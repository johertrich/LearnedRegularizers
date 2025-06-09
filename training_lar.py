# -*- coding: utf-8 -*-
"""
Training of Local (patch-based) adversarial regulariser.

@author: Alexander
"""

from priors import LocalAR
import torch
from deepinv.physics import Denoising, GaussianNoise
from training_methods import simple_lar_training
from deepinv.optim import L2
from dataset import get_dataset
from deepinv.datasets import PatchDataset

from torchvision.transforms import RandomCrop
from operators import get_operator

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "CT" # "Denoising" "CT"

# problem dependent parameters
if problem == "Denoising":
    physics , data_fidelity = get_operator(problem, device)
    dataset = get_dataset("BSDS500_gray", test=False, transform=RandomCrop(64))
    lmbd = 10.0
    test_ratio = 0.1
elif problem == "CT":
    dataset = get_dataset("LoDoPaB", test=False, transform=None)
    physics , data_fidelity = get_operator(problem, device)
    lmbd = 1.0
    test_ratio = 0.01
else:
    raise NotImplementedError

# splitting in training and validation set
test_len = int(len(dataset) * test_ratio)
train_len = len(dataset) - test_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])

batch_size = 256
shuffle = True

patch_size = 15

regularizer = LocalAR(in_channels=1, pad=False, use_bias=True, n_patches=-1).to(device)

mu = 5.0
simple_lar_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_set,
    val_set,
    patch_size=patch_size,
    device=device,
    epochs=400,
    lr=1e-4,
    mu=mu,
    batch_size=batch_size
)

dataset_name = "BSD500" if problem == "Denoising" else "LoDoPab"
torch.save(
    regularizer.cnn.state_dict(),
    f"weights/{regularizer.__class__.__name__}_adversarial_p={patch_size}x{patch_size}_{dataset_name}.pt",
)
