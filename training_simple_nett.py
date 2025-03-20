# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:30:12 2025

@author: JohannesSchwab
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

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "Denoising"
algorithm = "Adam"  # or "MAID", "Adam", "AdamW", "ISGD_Momentum"

# problem dependent parameters
if problem == "Tomography":
    noise_level = 0.1
    physics = Tomography(angles = 30, img_width = 128, circle = False)
    data_fidelity = L2(sigma=1.0)
    # Set the transform to a deterministic one like CenterCrop or Resize for MAID
    NT = NETT_transform(0.5,physics)
    resize_trans = Resize((128,128))
    tran = transforms.Compose([resize_trans,NT])
    dataset = get_dataset("BSDS500_gray", test=False, transform=tran)
    lmbd = 1.0


if problem == "Denoising":
    noise_level = 0.1
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
    data_fidelity = L2(sigma=1.0)
    NT = NETT_transform(0.5,physics)
    tran = transforms.Compose([CenterCrop(256),NT])
    dataset = get_dataset("BSDS500_gray", test=False, transform=tran)
    lmbd = 1



# splitting in training and validation set
test_ratio = 0.1
test_len = int(len(dataset) * 0.1)
train_len = len(dataset) - test_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])
batch_size = 4
shuffle = True

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=8, shuffle=True, drop_last=True
)

# define regularizer
regularizer = NETT(in_channels=1,out_channels = 1).to(device)

regularizer = simple_NETT_training(
    regularizer,
    train_dataloader,
    val_dataloader,
    device=device,
    optimizer=algorithm,
    lr=1e-3,
    num_epochs = 500
)
torch.save(regularizer.state_dict(), "weights/simple_NETT_denoising.pt")
