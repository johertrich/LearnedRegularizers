# -*- coding: utf-8 -*-
"""
Training of Local (patch-based) adversarial regulariser.

@author: Alexander
"""

from priors import LocalAR
import torch
from deepinv.physics import Denoising, GaussianNoise
from deepinv.optim import L2
from dataset import get_dataset
from deepinv.datasets import PatchDataset

from torchvision.transforms import RandomCrop
from operators import get_operator

from training_methods.bilevel_training import bilevel_training

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
    lmbd = 1.0
    test_ratio = 0.1
elif problem == "CT":
    dataset = get_dataset("LoDoPaB", test=False, transform=None)
    physics, data_fidelity = get_operator(problem, device)
    lmbd = 1.0
    test_ratio = 0.003
else:
    raise NotImplementedError

# splitting in training and validation set
test_len = int(len(dataset) * test_ratio)
train_len = len(dataset) - test_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])

batch_size = 2
shuffle = True
mode = "IFT" # "IFT" "JFB"

patch_size = 15

regularizer = LocalAR(in_channels=1, pad=True, use_bias=False, n_patches=-1, normalise_grad=False, pretrained="weights/LocalAR_bilevel_JFB_p=15x15_LoDoPab.pt").to(device)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=1)

dataset_name = "BSD500" if problem == "Denoising" else "LoDoPab"

regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=100,
    mode=mode,
    NAG_step_size=1e-1,
    NAG_max_iter=80, #500,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    minres_max_iter=10, #1000,
    minres_tol=1e-6,
    jfb_step_size_factor=1.0,
    lr=0.005,
    lr_decay=0.99,
    reg=False,
    reg_para=1e-5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
    validation_epochs=1,
    logger=None,
    dynamic_range_psnr=True if problem == "CT" else False,
    savestr=f"LocalAR_bilevel_{dataset_name}_{mode}",
    upper_loss=lambda x, y: torch.sum(((x - y) ** 2).view(x.shape[0], -1), -1),
)

torch.save(
    regularizer.cnn.state_dict(),
    f"weights/{regularizer.__class__.__name__}_bilevel_{mode}_p={patch_size}x{patch_size}_{dataset_name}.pt",
)
