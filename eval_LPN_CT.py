import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from deepinv.loss.metric import PSNR
from deepinv.utils.plotting import plot
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation import evaluate
from operators import get_evaluation_setting

# from deepinv.optim.optimizers import optim_builder
from priors.lpn.my_deepinv.deepinv_optimizers import optim_builder

# --- Accelerate setup ---
accelerator = Accelerator()
device = accelerator.device
print("device: ", device)
torch.random.manual_seed(0)  # make results deterministic

############################################################

# Problem selection
problem = "CT"  # Select problem setups, which we consider.

############################################################

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_path", type=str, default="lodopab")
parser.add_argument("--stepsize", type=float, default=0.01)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--max_iter", type=int, default=20)
parser.add_argument(
    "--save_psnr_path", type=str, default=None, help="Path to save PSNR results"
)
parser.add_argument(
    "--iterator", type=str, default="ADMM", help="Reconstruction algorithm"
)
parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
parser.add_argument(
    "--clamp", action="store_true", help="Clamp the output to [0, 1] at each iteration"
)
parser.add_argument("--lpn_no_patch", action="store_true")
parser.add_argument("--initialization", type=str, default="A_dagger")
parser.add_argument("--stride_size", type=int, default=None)
parser.add_argument("--exact_prox", action="store_true")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--verbose", action="store_true")
args = parser.parse_args()

stepsize = args.stepsize
max_iter = args.max_iter
batch_size = args.batch_size

#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################


# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)
if args.sample_ids:
    print(args.sample_ids)
    dataset = torch.utils.data.Subset(dataset, args.sample_ids)


angles = int(physics.radon.theta.shape[0])
noise_level_img = float(physics.noise_model.sigma.item())
print(f"Problem: {problem} | Angles: {angles} | Noise level: {noise_level_img}")


#############################################################
# Reconstruction algorithm
#############################################################

# Define regularizer
pretrained_path = args.pretrained_path
if args.lpn_no_patch:
    from priors.lpn.lpn_no_patch import LPNPrior

    regularizer = LPNPrior(pretrained=pretrained_path, clip=args.clamp).to(device)
else:
    from priors.lpn.lpn import LPNPrior

    regularizer = LPNPrior(
        pretrained=pretrained_path,
        clip=args.clamp,
        stride_size=args.stride_size,
        exact_prox=args.exact_prox,
    ).to(device)


iterator = args.iterator  # "ADMM"  # "PGD"
params_algo = {"stepsize": stepsize, "g_param": None, "beta": args.beta}
# Note:
# When iterator is "ADMM", stepsize here is "gamma" in the ADMM algorithm definition in https://deepinv.github.io/deepinv/user_guide/reconstruction/optimization.html#predefined-algorithms
# and "gamma" is 1/rho in Alg. 1 of https://arxiv.org/pdf/2310.14344

prior = regularizer
if args.initialization == "A_dagger":
    custom_init = lambda y, physics: {"est": (physics.A_dagger(y), physics.A_dagger(y))}
elif args.initialization == "zeros":
    custom_init = lambda y, physics: {
        "est": (
            torch.zeros_like(physics.A_dagger(y)),
            torch.zeros_like(physics.A_dagger(y)),
        )
    }
else:
    raise ValueError("Unknown initialization method", args.initialization)
model = optim_builder(
    iteration=iterator,
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=False,
    max_iter=max_iter,
    verbose=args.verbose,
    params_algo=params_algo,
    custom_init=custom_init,
)
model.eval()


def to_metric_tensor(x, device):
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        t = x
    elif isinstance(x, (float, int, bool)):
        t = torch.tensor(x)
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, (list, tuple)):
        # works for list of numbers or list of list of numbers (rectangular)
        t = torch.tensor(x)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

    if t.is_sparse:
        t = t.to_dense()
    return t.to(device)


def to_numpy(x):
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (float, int, bool)):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (list, tuple)):
        return np.array([to_numpy(item) for item in x])
    else:
        raise TypeError(f"Unsupported type: {type(x)}")


#############################################################
# Evaluation pipeline
#############################################################
def evaluate(
    physics,
    data_fidelity,
    dataset,
    model: nn.Module,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path=None,
    save_png=False,
    batch_size=8,
):
    """
    model: Reconstruction model, e.g., the return of `deepinv.optim.optimizers.optim_builder`
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()

    ## Evaluate on the test set
    psnrs = []
    monitor_data = defaultdict(list)
    for i, x in enumerate(tqdm(dataloader)):
        x = x.to(device).to(torch.float32)
        y = physics(x)
        # print(x.min(), x.max())
        # print(physics.A_dagger(y).min(), physics.A_dagger(y).max())

        # run the model on the problem.
        with torch.no_grad():
            recon, metrics = model(
                y, physics, x_gt=x, compute_metrics=True
            )  # reconstruction with PnP algorithm
            # print(metrics)
            # print(metrics["psnr"])
            # print(metrics["residual"])

        psnrs_batch = []
        for j in range(len(recon)):
            psnrs_batch.append(
                PSNR(max_pixel=None)(recon[j], x[j]).squeeze().detach().cpu().item()
            )
        psnrs_batch = accelerator.gather_for_metrics(psnrs_batch)
        psnrs.extend(psnrs_batch)

        for key in metrics.keys():
            vals = accelerator.gather_for_metrics(
                to_metric_tensor(metrics[key], device)
            )
            monitor_data[key].extend(vals)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            for j in range(len(recon)):

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 6))

                ax1.imshow(x[j, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                ax1.axis("off")
                ax1.set_title("ground truth")

                ax2.imshow(recon[j, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                ax2.axis("off")
                ax2.set_title("reconstruction")

                ax3.imshow(y[j, 0].cpu().numpy(), cmap="gray")
                ax3.axis("off")
                ax3.set_title("measurements")

                fig.suptitle(
                    f"IDX={i}-{j} | PSNR {np.round(psnrs[i*batch_size + j],3)}"
                )
                plt.savefig(os.path.join(save_path, f"imgs_{i}_{j}.png"))
                plt.close()

        if save_png and save_path is not None:
            for j in range(len(recon)):

                # Scale to [0, 255] and convert to uint8
                image_uint8 = (recon[j, 0].cpu().numpy().clip(0, 1) * 255).astype(
                    np.uint8
                )

                # Create a PIL Image object
                image = Image.fromarray(image_uint8, mode="L")  # 'L' for grayscale

                # Save as a PNG file
                image.save(os.path.join(save_path, f"reco_{i}_{j}.png"))

        if i == 0:
            y_out = y
            x_out = x
            recon_out = recon

    accelerator.wait_for_everyone()

    for key in monitor_data.keys():
        monitor_data[key] = to_numpy(monitor_data[key])

    mean_psnr = np.mean(psnrs)
    accelerator.print("Number of images: ", len(psnrs))
    accelerator.print("per iter psnr shape:", to_numpy(monitor_data["psnr"]).shape)
    accelerator.print("Mean PSNR over the test set: {0:.2f}".format(mean_psnr))
    accelerator.print(
        "Mean PSNR over the test set last step: {0:.2f}".format(
            np.mean(monitor_data["psnr"], axis=0)[-1]
        )
    )
    accelerator.print(
        "Mean PSNR over the test set best step: {0:.2f}, at step {1}".format(
            np.max(np.mean(monitor_data["psnr"], axis=0)),
            np.argmax(np.mean(monitor_data["psnr"], axis=0)),
        )
    )
    # print(recon_out.abs().max())
    return mean_psnr, x_out, y_out, recon_out, monitor_data


# Call unified evaluation routine
mean_psnr, x_out, y_out, recon_out, monitor_data = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
    model=model,
    device=device,
    save_png=True,
    save_path="imgs",
    batch_size=batch_size,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])

if args.save_psnr_path and accelerator.is_main_process:
    os.makedirs(os.path.dirname(args.save_psnr_path), exist_ok=True)
    np.save(args.save_psnr_path, {k: np.array(v) for k, v in monitor_data.items()})

accelerator.wait_for_everyone()
accelerator.free_memory()
accelerator.end_training()
