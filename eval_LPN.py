"""Evaluation script for LPN."""

import argparse
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from deepinv.loss.metric import PSNR
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.prior import PnP
from deepinv.utils.plotting import plot
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation import evaluate
from operators import get_evaluation_setting
from priors.lpn.lpn import LPNPrior

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device: ", device)
torch.random.manual_seed(0)  # make results deterministic


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    default="denoising",
    help="""\
- 'denoising' for denoising on BSD
- 'ct_trained_on_bsd' for CT with model trained on BSD
- 'ct' for CT with model trained on LoDoPaB""",
)
parser.add_argument("--pretrained_path", type=str, default=None)
##############################################
# ADMM parameters in CT reconstruction
parser.add_argument("--stepsize", type=float, default=None)
parser.add_argument("--beta", type=float, default=None)
parser.add_argument("--max_iter", type=int, default=None)
##############################################
parser.add_argument(
    "--gamma",
    type=float,
    default=None,
    help="gamma in the Identity residual connection. gamma=1 is the original denoiser.",
)
args = parser.parse_args()


############################################################
# Select default parameters
task = args.task
if task == "denoising":
    problem = "Denoising"
    pretrained_path = args.pretrained_path or "weights/lpn_64_bsd/LPN.pt"
elif task == "ct_trained_on_bsd":
    problem = "CT"
    pretrained_path = args.pretrained_path or "weights/lpn_64_bsd/LPN.pt"
    stepsize = args.stepsize or 0.015
    beta = args.beta or 1.0
    max_iter = args.max_iter or 20
    gamma = args.gamma or 0.9
elif task == "ct":
    problem = "CT"
    pretrained_path = args.pretrained_path or "weights/lpn_64_ct/LPN.pt"
    stepsize = args.stepsize or 0.02
    beta = args.beta or 1.0
    max_iter = args.max_iter or 20
    gamma = args.gamma or 1.0
else:
    raise ValueError("Unknown task. Choose 'denoising', 'ct_trained_on_bsd' or 'ct'.")

only_first = False  # just evaluate on the first image of the dataset for test purposes

############################################################


#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)
if problem == "CT":
    angles = int(physics.radon.theta.shape[0])
    noise_level_img = float(physics.noise_model.sigma.item())
    print(f"Problem: {problem} | Angles: {angles} | Noise level: {noise_level_img}")

if problem == "Denoising":
    adaptive_range = False
else:
    adaptive_range = True

#############################################################
# Reconstruction algorithm
#############################################################

# Define regularizer
regularizer = LPNPrior(pretrained=pretrained_path).to(device)
regularizer.eval()


if task in ["ct_trained_on_bsd", "ct"]:
    # Use PnP-ADMM for CT reconstruction
    iterator = "ADMM"
    params_algo = {"stepsize": stepsize, "g_param": None, "beta": beta}

    if gamma is None or gamma == 1.0:
        denoiser = regularizer.prox
    else:
        # Apply the Identity residual connection
        def denoiser(x, *args, **kwargs):
            return gamma * regularizer.prox(x) + (1 - gamma) * x

    model = optim_builder(
        iteration=iterator,
        prior=PnP(denoiser=denoiser),
        data_fidelity=data_fidelity,
        early_stop=True,
        max_iter=max_iter,
        verbose=True,
        params_algo=params_algo,
        custom_init=lambda y, physics: {
            "est": (physics.A_dagger(y), physics.A_dagger(y))
        },
    )
    model.eval()
else:
    # For denoising, call the learned proximal operator directly without iterative optimization
    model = lambda y, physics: regularizer.prox(y)


#############################################################
# Evaluation pipeline
#############################################################
def evaluate(
    physics,
    data_fidelity,
    dataset,
    model: Callable,
    only_first=False,
    adaptive_range=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path=None,
    save_png=False,
):
    """
    model: Callable: y, physics -> recon. Reconstruction algorithm.
    """

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    if adaptive_range:
        psnr = PSNR(max_pixel=None)
    else:
        psnr = PSNR()

    ## Evaluate on the test set
    psnrs = []
    iters = []
    Lip = []
    for i, x in enumerate(tqdm(dataloader)):
        if device == "mps":
            # mps does not support float64
            x = x.to(torch.float32).to(device)
        else:
            x = x.to(device).to(torch.float32)
        y = physics(x)

        # run the model on the problem.
        with torch.no_grad():
            recon = model(y, physics)

        psnrs.append(psnr(recon, x).squeeze().item())
        print(f"Image {i:03d} | PSNR: {psnrs[-1]:.2f}")

        if save_path is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 6))

            ax1.imshow(x[0, 0].cpu().numpy(), cmap="gray")
            ax1.axis("off")
            ax1.set_title("ground truth")

            ax2.imshow(recon[0, 0].cpu().numpy(), cmap="gray")
            ax2.axis("off")
            ax2.set_title("reconstruction")

            ax3.imshow(y[0, 0].cpu().numpy(), cmap="gray")
            ax3.axis("off")
            ax3.set_title("measurements")

            fig.suptitle(f"IDX={i} | PSNR {np.round(psnrs[-1],3)}")
            plt.savefig(os.path.join(save_path, f"imgs_{i}.png"))
            plt.close()

        if save_png and save_path is not None:

            # Scale to [0, 255] and convert to uint8
            image_uint8 = (recon[0, 0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

            # Create a PIL Image object
            image = Image.fromarray(image_uint8, mode="L")  # 'L' for grayscale

            # Save as a PNG file
            image.save(os.path.join(save_path, f"reco_{i}.png"))

        if i == 0:
            y_out = y
            x_out = x
            recon_out = recon
        if only_first:
            break
    mean_psnr = np.mean(psnrs)
    print("Mean PSNR over the test set: {0:.2f}".format(mean_psnr))
    # mean_iters = np.mean(iters)
    # print("Mean iterations over the test set: {0:.2f}".format(mean_iters))
    # mean_Lip = np.mean(Lip)
    # print("Mean L over the test set: {0:.2f}".format(mean_Lip))
    return mean_psnr, x_out, y_out, recon_out


# Call unified evaluation routine
mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
    model=model,
    only_first=only_first,
    adaptive_range=adaptive_range,
    device=device,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
