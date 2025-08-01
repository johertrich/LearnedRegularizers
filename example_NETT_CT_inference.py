# -*- coding: utf-8 -*-
"""
Created on Tue May  6 20:47:24 2025

@author: Johannes
"""



from deepinv.physics import Denoising, MRI, GaussianNoise, Tomography
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from evaluation import evaluate
from dataset import get_dataset
from operators import get_operator, get_evaluation_setting
from torchvision.transforms import CenterCrop
from priors import NETT
import torch
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device: ", device)
torch.random.manual_seed(0)  # make results deterministic

############################################################

# Problem selection

problem = "CT"  # Select problem setups, which we consider.
only_first = False  # just evaluate on the first image of the dataset for test purposes

############################################################

# Define regularizer


regularizer = NETT(in_channels = 1, out_channels = 1).to(device)
regularizer.positivity = True
regularizer.load_state_dict(torch.load('weights/NETT_weights_CT.pt'))
regularizer.eval()

# reconstruction hyperparameters, might be problem dependent
if problem == "Denoising":
    lmbd = 3
    # regularization parameter
elif problem == "CT":
    lmbd = 600.0  # 40 for CT_training # regularization parameter

# Parameters for the Nesterov Algorithm, might also be problem dependent...

NAG_step_size = 1e-1  # step size in NAG
NAG_max_iter = 400  # maximum number of iterations in NAG
NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)


#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator
if problem == "Denoising":
    noise_level = 0.1
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
    data_fidelity = L2(sigma=1.0)
    dataset = get_dataset("BSD68", transform=CenterCrop(320))
elif problem == "CT":
    dataset, physics, data_fidelity = get_evaluation_setting(problem, device)
else:
    raise NotImplementedError("Problem not found")

#regularizer.compute_padding((dataset[torch.tensor(0)][0].shape[-2],dataset[torch.tensor(0)][0].shape[-1]))


# Call unified evaluation routine

mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
    regularizer=regularizer,
    lmbd=lmbd,
    NAG_step_size=NAG_step_size,
    NAG_max_iter=NAG_max_iter,
    NAG_tol=NAG_tol,
    only_first=only_first,
    adaptive_range = True,
    device=device,
    verbose=True,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out],cbar = True)

#reg_denoised = regularizer(y_out)
