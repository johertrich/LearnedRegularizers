# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 08:51:30 2025

@author: Johannes
"""

from priors import NETT
import torch
from deepinv.physics import Denoising, GaussianNoise, Tomography
from training_methods import simple_NETT_training
from deepinv.optim import L2
from dataset import get_dataset
from dataset.utils import NETT_transform
from torchvision.transforms import RandomCrop, CenterCrop, Resize
from torch.utils.data import Subset as subset
from torchvision import transforms
import matplotlib.pyplot as plt

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "Denoising"
algorithm = "Adam"  # or "MAID", "Adam", "AdamW", "ISGD_Momentum"

n_iter = 10000
regularization_param = 0.001
lam = 0.0005
sample_idx = 55

# problem dependent parameters
if problem == "Tomography":
    noise_level = 0.1
    physics = Tomography(angles = 30, img_width = 128, circle = False)
    NT = NETT_transform(0.0,physics)
    resize_trans = Resize((128,128))
    tran = transforms.Compose([resize_trans,NT])
    dataset = get_dataset("BSDS500_gray", test=True, transform=tran)

if problem == "Denoising":
    noise_level = 0.1
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
    data_fidelity = L2(sigma=1.0)
    NT = NETT_transform(0.0,physics)
    tran = transforms.Compose([RandomCrop(256),NT])
    dataset = get_dataset("BSDS500_gray", test=False, transform=tran)
    lmbd = 1


# choose random image and generate data

image = dataset[sample_idx][0].unsqueeze(1).to(device)
data = physics(image)

# define regularizer

regularizer = NETT(in_channels=1,out_channels = 1).to(device)
regularizer.load_state_dict(torch.load('weights/simple_NETT_denoising.pt'))

# Initialize with zero

x = torch.zeros_like(image,requires_grad = True)
x_land = torch.zeros_like(image,requires_grad = True)
error = []
error_land = []
reg_value = []

# Alternating gradient descent for the NETT functional

for i in range(n_iter):
    x = x.cpu()-lam*physics.A_adjoint(physics.A(x).cpu()-data.cpu())
    x_land = x_land.cpu()-lam*physics.A_adjoint(physics.A(x_land).cpu()-data.cpu())
    if i%100 == 0 and i>0:
        plt.subplot(221)
        plt.imshow(x_land[0,0].detach().cpu())
        plt.subplot(222)
        plt.imshow(x[0,0].detach())
        plt.title('iteration '+str(i))
        plt.subplot(223)
        plt.plot(error[28:], label = 'NETT')
        plt.plot(error_land[28:], label = 'Landweber')
        plt.legend()
        plt.subplot(224)
        plt.plot(reg_value, label = 'regularization value')
        plt.legend()
        plt.show()

    out = regularizer.regularizer(x.to(device))
    grad = torch.autograd.grad(outputs=out, inputs=x, retain_graph=True)[0]
    x = x - regularization_param*grad
    error.append(torch.mean((x.detach().cpu()-image.cpu())**2))
    error_land.append(torch.mean((x_land.detach().cpu()-image.cpu())**2))
    reg_value.append(out.detach().cpu())
    

