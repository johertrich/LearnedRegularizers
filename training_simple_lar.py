# -*- coding: utf-8 -*-
"""
Training of Local (patch-based) adversarial regulariser.

@author: Alexander
"""

from priors import LocalAR
import torch
from deepinv.physics import Denoising, GaussianNoise, Tomography
from training_methods import simple_ar_training
from training_methods.simple_ar_training import estimate_lmbd
from deepinv.optim import L2
from dataset import get_dataset
from deepinv.datasets import PatchDataset

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
    dataset = get_dataset("BSDS500_gray", test=False, transform=RandomCrop(300))
    lmbd = 1

# splitting in training and validation set
test_ratio = 0.1
test_len = int(len(dataset) * 0.1)
train_len = len(dataset) - test_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])
batch_size = 512
shuffle = True

patch_size = 15

train_imgs = torch.stack([train_set[i] for i in range(len(train_set))])
#train_imgs = torch.stack([train_set[i] for i in range(5)])

val_imgs = torch.stack([val_set[i] for i in range(len(val_set))])
#val_imgs = torch.stack([val_set[i] for i in range(2)])

print(train_imgs.shape, val_imgs.shape)

train_dataset = PatchDataset(train_imgs, patch_size=patch_size, shapes=(1, patch_size, patch_size), transforms=None)
val_dataset = PatchDataset(val_imgs, patch_size=patch_size, shapes=(1, patch_size, patch_size), transforms=None)

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)

lmbd = 1.0 #estimate_lmbd(train_dataloader,physics,device)

print("Esimated lambda: ", lmbd)

# define regularizer
# regularizer = ICNNPrior(
#     in_channels=1, strong_convexity=0, num_layers=5, num_filters=16
# ).to(device)
# regularizer = linearICNNPrior(
#     in_channels=1, strong_convexity=0, num_layers=5, num_filters=16
# ).to(device)
regularizer = LocalAR(
    in_channels=1
).to(device)

mu = 10.0
simple_ar_training(
    regularizer.cnn,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    device=device,
    epochs=10,
    lr=1e-4,
    mu=mu
)

torch.save(regularizer.cnn.state_dict(), f"weights/simple_{regularizer.__class__.__name__}_mu={mu}_local_ar.pt")
