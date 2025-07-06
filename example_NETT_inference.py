# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 17:17:40 2025

@author: Johannes
"""

from deepinv.physics import Denoising, MRI, GaussianNoise, Tomography
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from operators import get_operator, get_evaluation_setting
from evaluation import evaluate
from dataset import get_dataset
from torchvision.transforms import CenterCrop
from priors import ICNNPrior, NETT, DRUNet, simpleNETT
import torch

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

problem = "Denoising"  # Select problem setups, which we consider.
only_first = False  # just evaluate on the first image of the dataset for test purposes

############################################################

# Define regularizer

regularizer = simpleNETT(in_channels = 1, out_channels = 1).to(device)

#regularizer = DRUNet(in_channels=1,out_channels = 1,device = device,p = 2)
regularizer.load_state_dict(torch.load('weights/NETT_weights_denoising.pt'))


# reconstruction hyperparameters, might be problem dependent
if problem == "Denoising":
    lmbd = 6
    
    # regularization parameter
elif problem == "CT":
    lmbd = 500.0  # regularization parameter

# Parameters for the Nesterov Algorithm, might also be problem dependent...

NAG_step_size = 1e-1  # step size in NAG
NAG_max_iter = 800  # maximum number of iterations in NAG
NAG_tol = 1e-6  # tolerance for the relative error (stopping criterion)


#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator
if problem == "Denoising":
    dataset, physics, data_fidelity = get_evaluation_setting(problem, device)


# Call unified evaluation routine

#regularizer.compute_padding((dataset[0][0].shape[-2],dataset[0][0].shape[-1]))

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
    device=device,
    verbose=True,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])

reg_denoised = regularizer(y_out)
