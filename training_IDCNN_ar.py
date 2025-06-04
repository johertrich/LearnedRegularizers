# -*- coding: utf-8 -*-
"""
Created on May 21 2025

@author: Yasi Zhang
"""

from priors import ICNNPrior, CNNPrior,linearICNNPrior, WCRR, simple_ICNNPrior, linearIDCNNPrior
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
from operators import get_operator

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "CT"

# problem dependent parameters
physics , data_fidelity = get_operator(problem, device)
lmbd = 1.0
noise_level = 0.1

train_dataset = get_dataset("LoDoPaB", test=False, transform=None)
# splitting in training and validation set
test_ratio = 0.1
val_len = int(len(train_dataset) * 0.1)
train_len = len(train_dataset) - val_len
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = torch.utils.data.Subset(train_dataset, range(train_len, len(train_dataset)))


# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=32, shuffle=True, drop_last=True, num_workers=8
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True, num_workers=8
)

lmbd = estimate_lmbd(train_dataloader,physics,device)


regularizer = linearIDCNNPrior(in_channels=1,
    num_filters=32,
    kernel_dim=5,
    num_layers=5,
).to(device)

idx=1

simple_ar_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    device=device,
    epochs=10,
    lr=1e-3,
    mu=10,
    idx=idx,
)

torch.save(regularizer.state_dict(), f"weights/simple_{regularizer.__class__.__name__}_ar_{problem}.pt")


