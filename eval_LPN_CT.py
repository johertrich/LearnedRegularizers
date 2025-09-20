import argparse

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from deepinv.loss.metric import PSNR
from deepinv.optim.optimizers import optim_builder
from deepinv.utils.plotting import plot
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation import evaluate
from operators import get_evaluation_setting
from priors.lpn.lpn import LPNPrior

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
parser.add_argument("--pretrained_path", type=str)
parser.add_argument("--stepsize", type=float, default=0.01)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--max_iter", type=int, default=100)
parser.add_argument("--sample_ids", nargs="+", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################


# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)
if args.sample_ids:
    print("Testing samples:", args.sample_ids)
    dataset = torch.utils.data.Subset(dataset, args.sample_ids)


angles = int(physics.radon.theta.shape[0])
noise_level_img = float(physics.noise_model.sigma.item())
print(f"Problem: {problem} | Angles: {angles} | Noise level: {noise_level_img}")


#############################################################
# Reconstruction algorithm
#############################################################
# Define regularizer
regularizer = LPNPrior(pretrained=args.pretrained_path, clip=True).to(device)

params_algo = {"stepsize": args.stepsize, "g_param": None, "beta": args.beta}
prior = regularizer
custom_init = lambda y, physics: {"est": (physics.A_dagger(y), physics.A_dagger(y))}
model = optim_builder(
    iteration="ADMM",
    prior=prior,
    data_fidelity=data_fidelity,
    early_stop=False,
    max_iter=args.max_iter,
    verbose=True,
    params_algo=params_algo,
    custom_init=custom_init,
)
model.eval()


#############################################################
# Evaluation pipeline
#############################################################
def evaluate(
    physics,
    data_fidelity,
    dataset,
    model: nn.Module,
    device="cuda" if torch.cuda.is_available() else "cpu",
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
    for i, x in enumerate(tqdm(dataloader)):
        x = x.to(device).to(torch.float32)
        y = physics(x)

        # run the model on the problem.
        with torch.no_grad():
            recon, metrics = model(
                y, physics, x_gt=x, compute_metrics=True
            )  # reconstruction with PnP algorithm

        psnrs_batch = []
        for j in range(len(recon)):
            psnrs_batch.append(
                PSNR(max_pixel=None)(recon[j], x[j]).squeeze().detach().cpu().item()
            )
        psnrs_batch = accelerator.gather_for_metrics(psnrs_batch)
        psnrs.extend(psnrs_batch)

        if i == 0:
            y_out = y
            x_out = x
            recon_out = recon

    accelerator.wait_for_everyone()

    mean_psnr = np.mean(psnrs)
    accelerator.print("Number of images: ", len(psnrs))
    accelerator.print("Mean PSNR over the test set: {0:.2f}".format(mean_psnr))
    return mean_psnr, x_out, y_out, recon_out


# Call unified evaluation routine
mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
    model=model,
    device=device,
    batch_size=args.batch_size,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
if accelerator.is_main_process:
    plot([x_out, y_out, recon_out], save_fn="tmp.png")

accelerator.wait_for_everyone()
accelerator.free_memory()
accelerator.end_training()
