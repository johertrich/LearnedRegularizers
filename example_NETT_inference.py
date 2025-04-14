# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 17:17:40 2025

@author: Johannes
"""

from deepinv.physics import Denoising, MRI, GaussianNoise, Tomography
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from evaluation import evaluate
from dataset import get_dataset
from operators import MRIonR
from torchvision.transforms import CenterCrop
from priors import ICNNPrior, NETT, DRUNet
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

regularizer = DRUNet(in_channels=1,out_channels = 1,device = device)
regularizer.load_state_dict(torch.load('weights/simple_DRUNETT_denoising.pt'))


# reconstruction hyperparameters, might be problem dependent
if problem == "Denoising":
    lmbd = 0.7
    # regularization parameter
elif problem == "CT":
    lmbd = 500.0  # regularization parameter

# Parameters for the Nesterov Algorithm, might also be problem dependent...

NAG_step_size = 1e-4  # step size in NAG
NAG_max_iter = 5000  # maximum number of iterations in NAG
NAG_tol = 1e-6  # tolerance for the relative error (stopping criterion)


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
    noise_level = 0.5
    dataset = get_dataset("BSD68", transform=CenterCrop(256))
    imsize = dataset[0].shape[-1]
    physics = Tomography(
        imsize // 3, imsize, device=device, noise_model=GaussianNoise(sigma=noise_level)
    )
    data_fidelity = L2(sigma=1.0)
# elif problem == "MRI":
#    dataset = get_dataset("BSDS500_gray", transform=CenterCrop(256), test=True)
#    img_size = dataset[0].shape
#    noise_level = 0.05
#    # simple Cartesian mask generation from the deepinv tour...
#    mask = torch.rand((1, img_size[-1]), device=device) > 0.75
#    mask = torch.ones((img_size[-2], 1), device=device) * mask
#    mask[:, int(img_size[-1] / 2) - 2 : int(img_size[-1] / 2) + 2] = 1
#    # The MRI operator in deepinv operates on complex-valued images.
#    # The MRIonR operator wraps it for real-valued images
#    physics = MRIonR(
#        mask=mask, device=device, noise_model=GaussianNoise(sigma=noise_level)
#    )
#    data_fidelity = L2(sigma=1.0)
else:
    raise NotImplementedError("Problem not found")

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
    device=device,
    verbose=True,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])

reg_denoised = regularizer(y_out)
