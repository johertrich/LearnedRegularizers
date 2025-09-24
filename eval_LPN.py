"""Evaluation script for LPN."""

import argparse
import datetime
import logging
import os
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from deepinv.loss.metric import PSNR
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.prior import PnP
from deepinv.utils.plotting import plot
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

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
    "--problem",
    type=str,
    default="Denoising",
)  # "Denoising" or "CT"
parser.add_argument("--dataset", type=str, default="BSD")
parser.add_argument("--pretrained_path", type=str, default=None)
##############################################
# ADMM parameters in CT reconstruction
parser.add_argument("--stepsize", type=float, default=None)
parser.add_argument("--beta", type=float, default=None)
parser.add_argument("--max_iter", type=int, default=None)
##############################################
parser.add_argument("--only_first", type=bool, default=False)
parser.add_argument("--save_results", type=bool, default=False)
args = parser.parse_args()


############################################################
# Select default parameters
problem = args.problem
if args.pretrained_path is None:
    if problem == "CT" and args.dataset == "BSD":
        pretrained_path = args.pretrained_path or "weights/lpn_64_BSD_noise_0.05/LPN.pt"
        stepsize = args.stepsize or 0.008
        beta = args.beta or 1.0
        max_iter = args.max_iter or 100
    elif problem == "CT" and args.dataset == "LoDoPaB":
        pretrained_path = args.pretrained_path or "weights/lpn_64_CT_noise_0.1/LPN.pt"
        stepsize = args.stepsize or 0.02
        beta = args.beta or 1.0
        max_iter = args.max_iter or 100
    elif problem == "Denoising":
        pretrained_path = "weights/lpn_64_BSD_noise_0.1/LPN.pt"
    else:
        raise ValueError(
            "Unknown task. Choose problem 'Denoising' and dataset 'BSD' or problem 'CT' and dataset in ['BSD', 'LoDoPaB']."
        )

only_first = (
    args.only_first
)  # just evaluate on the first image of the dataset for test purposes
save_results = args.save_results  # If True, save the first 10 image reconstructions

if save_results:
    save_path = f"savings/{problem}/LPN/{args.dataset}"
    logging_path = save_path + "/logging"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(logging_path, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logging_path
        + "/log_eval_"
        + problem
        + "_LPN_"
        + str(datetime.datetime.now())
        + ".log",
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
    )
    logger.info(f"Evaluation LPN on {problem}!!!")

############################################################

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
regularizer = LPNPrior(pretrained=pretrained_path, clip=True).to(device)
regularizer.eval()


if problem == "CT":
    # Use PnP-ADMM for CT reconstruction
    params_algo = {"stepsize": stepsize, "g_param": None, "beta": beta}

    model = optim_builder(
        iteration="ADMM",
        prior=regularizer,
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
    logger=None,
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
    times = []  # List of recon durations for each test image
    x_out = None
    y_out = None
    recon_out = None
    for i, x in (progress_bar := tqdm(enumerate(dataloader))):
        x = x.to(torch.float32).to(device)
        y = physics(x)

        t_start = time.time()

        # run the model on the problem.
        with torch.no_grad():
            recon = model(y, physics)

        t_end = time.time()
        times.append(t_end - t_start)
        psnrs.append(psnr(recon, x).squeeze().item())

        if logger is not None:
            logger.info(f"Image {i} reconstructed, PSNR: {psnrs[-1]:.2f}")

        if save_path is not None and (i < 10):
            save_image(x, os.path.join(save_path, f"ground_truth_{i}.png"), padding=0)
            save_image(y, os.path.join(save_path, f"measurement_{i}.png"), padding=0)
            save_image(
                recon, os.path.join(save_path, f"reconstruction_{i}.png"), padding=0
            )

        if i == 0:
            y_out = y
            x_out = x
            recon_out = recon

        progress_bar.set_description(
            f"Mean PSNR: {np.mean(psnrs):.2f}, Last PSNR: {psnrs[-1]:.2f}"
        )
        if only_first:
            break
    mean_psnr = np.mean(psnrs)
    print_psnr = "Mean PSNR over the test set: {0:.2f}".format(mean_psnr)
    print(print_psnr)
    mean_time = np.mean(times)
    print_time = "Mean reconstruction time over the test set: {0:.2f} seconds".format(
        mean_time
    )
    print(print_time)

    if logger is not None:
        logger.info(print_psnr)
        logger.info(print_time)

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
    save_path=save_path if save_results else None,
    logger=logger if save_results else None,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
