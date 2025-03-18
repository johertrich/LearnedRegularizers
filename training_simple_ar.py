# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 2025

@author: Zakobian
"""

from priors import ICNNPrior, CNNPrior,linearICNNPrior
import torch
from deepinv.physics import Denoising, GaussianNoise, Tomography
from training_methods import simple_ar_training
from training_methods.simple_ar_training import estimate_lmbd
from deepinv.optim import L2
from dataset import get_dataset
from torchvision.transforms import RandomCrop, CenterCrop, Resize
from torch.utils.data import Subset as subset
from torchvision import transforms
import numpy as np

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# problem = "Tomography"
problem = "Denoising"

# problem dependent parameters
if problem == "Denoising":
    noise_level = 0.1
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
    data_fidelity = L2(sigma=1.0)
    # dataset = get_dataset("BSDS500_gray", test=False, transform=RandomCrop(64))
    dataset = get_dataset("BSDS500_gray", test=False, transform=RandomCrop(64))
    lmbd = 1

# splitting in training and validation set
test_ratio = 0.1
test_len = int(len(dataset) * 0.1)
train_len = len(dataset) - test_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])
batch_size = 8
shuffle = True

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=8, shuffle=True, drop_last=True
)

lmbd = estimate_lmbd(train_dataloader,physics,device)

# define regularizer
# regularizer = ICNNPrior(
#     in_channels=1, strong_convexity=0, num_layers=5, num_filters=16
# ).to(device)
# regularizer = linearICNNPrior(
#     in_channels=1, strong_convexity=0, num_layers=5, num_filters=16
# ).to(device)
regularizer = CNNPrior(
    in_channels=1, size=64
).to(device)

simple_ar_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    device=device,
)

torch.save(regularizer.state_dict(), f"weights/simple_{regularizer.__class__.__name__}_ar.pt")
