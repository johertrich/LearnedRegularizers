# -*- coding: utf-8 -*-
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from priors import WCRR, ParameterLearningWrapper
import torch
from deepinv.physics import Denoising, GaussianNoise
from training_methods import simple_ar_training, bilevel_training
from training_methods.simple_ar_training import estimate_lmbd, estimate_lip
from deepinv.optim import L2
from dataset import get_dataset
from torchvision.transforms import RandomCrop, CenterCrop, Compose
from operators.settings import get_operator
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

problem = "Denoising"

# problem dependent parameters
transform = Compose(
    [
    RandomCrop(64),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomApply([RandomRotation((90, 90))], p=0.5),
    ]
)
dataset = get_dataset("BSDS500_gray", test=False, transform=transform)
physics, data_fidelity = get_operator(problem, device)

# splitting in training and validation set
test_ratio = 0.1
test_len = int(len(dataset) * 0.1)
train_len = len(dataset) - test_len
train_set = torch.utils.data.Subset(dataset, range(train_len))
val_dataset = get_dataset("BSDS500_gray", test=False, transform=CenterCrop(321))
val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(dataset)))
batch_size = 8
shuffle = True

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=True, drop_last=True
)

regularizer = WCRR(
    sigma=0.1,
    weak_convexity=0.0,
    ).to(device)

only_fitting = False
if only_fitting:
    ckp = torch.load(f"weights/adversarial_{problem}/CRR_adversarial_for_{problem}.pt")
    regularizer.load_state_dict(ckp)
else:
    regulalrizer = simple_ar_training(
        regularizer,
        physics,
        data_fidelity,
        train_dataloader,
        val_dataloader,
        device=device,
        epochs=500,
        validation_epochs=50,
        lr=1e-2,
        lr_decay = 0.995,
        mu=10.0,
    )
    torch.save(regularizer.state_dict(), f"weights/adversarial_{problem}/CRR_adversarial_for_{problem}.pt")

lmbd = estimate_lmbd(val_dataloader,physics,device)
lip = estimate_lip(regularizer,val_dataloader,device)
lmbd_est = lmbd/lip
print("Estimated Lambda for Fitting:", lmbd_est)

wrapped_regularizer = ParameterLearningWrapper(regularizer, device=device)
wrapped_regularizer.alpha.data = torch.log(lmbd_est)
for p in wrapped_regularizer.parameters():
    p.requires_grad_(False)
    wrapped_regularizer.alpha.requires_grad_(True)
    wrapped_regularizer.scale.requires_grad_(True)

# parameter search
bilevel_training(
    wrapped_regularizer,
    physics,
    data_fidelity,
    1,
    val_dataloader,
    val_dataloader,
    epochs=10,
    mode='IFT',
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    lr=0.1,
    lr_decay=0.95,
    device=device,
    verbose=False,
    validation_epochs=10,
    dynamic_range_psnr=True,
)

torch.save(wrapped_regularizer.state_dict(), f"weights/adversarial_{problem}/CRR_adversarial_for_{problem}_fitted.pt")

print("Final alpha: ", torch.exp(wrapped_regularizer.alpha))
print("Final scale: ", wrapped_regularizer.scale)
