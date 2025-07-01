# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 2025

@author: Zakobian
"""
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from priors import ICNNPrior, CNNPrior,linearICNNPrior, WCRR, simple_ICNNPrior
import torch
from deepinv.physics import Denoising, GaussianNoise, Tomography
from training_methods import simple_ar_training
from training_methods.simple_ar_training import estimate_lmbd
from deepinv.optim import L2
from dataset import get_dataset
from torchvision.transforms import RandomCrop, CenterCrop, Compose

from torch.utils.data import Subset as subset
from torchvision import transforms
import numpy as np
from torchvision.transforms import (
    RandomCrop,
    RandomVerticalFlip,
    Compose,
    RandomHorizontalFlip,
    CenterCrop,
    RandomApply,
    RandomRotation,
)
if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# problem = "CT"
problem = "Denoising"

# problem dependent parameters
if problem == "Denoising":
    transform = Compose(
    [
        RandomCrop(64),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomApply([RandomRotation((90, 90))], p=0.5),
    ]
    )
    dataset = get_dataset("BSDS500_gray", test=False, transform=transform)

# if problem == "CT":
    # dataset = get_dataset("LoDoPaB", test=False)

# def get_evaluation_setting(problem, device):
#     physics, data_fidelity = get_operator(problem, device)
#     if problem == "Denoising":
#         dataset = get_dataset("BSD68")
#     elif problem == "CT":
#         dataset = get_dataset("LoDoPaB", test=True)
#     return dataset, physics, data_fidelity

from operators.settings import get_operator
physics, data_fidelity = get_operator(problem, device)

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
    val_set, batch_size=1, shuffle=True, drop_last=True
)

lmbd = estimate_lmbd(train_dataloader,physics,device)
regularizer = simple_ICNNPrior(in_channels=1,channels=32,device=device)


simple_ar_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    device=device,
    epochs=1000,
    validation_epochs=100,
    lr=1e-3,
    lr_decay = 0.998,
    mu=10.0,
)
torch.save(regularizer.state_dict(), f"weights/{regularizer.__class__.__name__}_ar_{problem}.pt")
