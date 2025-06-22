"""Train LPN by prox matching on CT."""

import torch
import numpy as np
from deepinv.physics import Denoising, GaussianNoise
from deepinv.optim import L2
from dataset import get_dataset
from torchvision.transforms import (
    RandomCrop,
    RandomVerticalFlip,
    Compose,
    RandomHorizontalFlip,
    CenterCrop,
    RandomApply,
    RandomRotation,
)
from PIL import Image
import os
import matplotlib.pyplot as plt

from priors.lpn.lpn import LPNPrior
from training_methods.simple_lpn_training import simple_lpn_training, Validator

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "CT"
crop_size = 64

# problem dependent parameters
noise_level = 0.1
physics = None
data_fidelity = None

transform = Compose(
    [
        RandomCrop(crop_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomApply([RandomRotation((90, 90))], p=0.5),
    ]
)
train_dataset = get_dataset("LoDoPaB", test=False, transform=transform)
test_transform = None  # CenterCrop(crop_size)
val_dataset = get_dataset("LoDoPaB", test=False, transform=test_transform)
# split train and val
val_ratio = 0.1
val_len = int(len(train_dataset) * val_ratio)
train_len = len(train_dataset) - val_len
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))


# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=128, shuffle=True, drop_last=True, num_workers=8
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True, num_workers=8
)
data = next(iter(train_dataloader))
print(data.shape, data.min(), data.max())


###############################################################################
# Training
###############################################################################
# define regularizer
regularizer = LPNPrior(model_name="lpn_64_neg1").to(device)

ckpt_dir = "weights/lpn_64_ct"
os.makedirs(ckpt_dir, exist_ok=True)
regularizer = simple_lpn_training(
    regularizer=regularizer,
    physics=None,
    data_fidelity=None,
    lmbd=None,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    device=device,
    sigma_noise=noise_level,
    num_steps=320_000,
    num_steps_pretrain=160_000,
    validate_every_n_steps=10_000,
    ckpt_dir=ckpt_dir,
    loss_type="pm",
)
torch.save(regularizer.state_dict(), f"{ckpt_dir}/simple_LPN.pt")
