# -*- coding: utf-8 -*-
import logging
import datetime
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from priors import (
    LSR,
    WCRR,
    ICNNPrior,
    IDCNNPrior,
    TDV,
    ParameterLearningWrapper,
    LocalAR,
)
import torch
from training_methods import ar_training, bilevel_training
from training_methods.ar_training import estimate_lmbd, estimate_lip
from dataset import get_dataset
from operators.settings import get_operator
from torch.utils.data import Subset as subset
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
from hyperparameters import get_AR_hyperparameters
import argparse

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser(description="Choosing training setting")
parser.add_argument("--problem", type=str, default="Denoising")
parser.add_argument("--regularizer_name", type=str, default="TDV")
parser.add_argument("--only_fitting", type=str, default=False)
inp = parser.parse_args()

problem = inp.problem  # Denoising or CT
regularizer_name = inp.regularizer_name  # CRR, WCRR, ICNN, IDCNN, TDV or LSR
only_fitting = inp.only_fitting  # True or False

hyper_params = get_AR_hyperparameters(regularizer_name, problem)


# define regularizer
if regularizer_name == "CRR":
    regularizer = WCRR(
        sigma=0.1,
        weak_convexity=0.0,
    ).to(device)
elif regularizer_name == "WCRR":
    regularizer = WCRR(
        sigma=0.1,
        weak_convexity=1.0,
    ).to(device)
elif regularizer_name == "ICNN":
    regularizer = ICNNPrior(in_channels=1, channels=32, device=device)
elif regularizer_name == "IDCNN":
    regularizer = IDCNNPrior(in_channels=1, channels=32, kernel_size=5, device=device)
elif regularizer_name == "TDV":
    config = dict(
        in_channels=1,
        num_features=32,
        multiplier=1,
        num_mb=3,
        num_scales=3,
        zero_mean=True,
    )
    regularizer = TDV(**config).to(device)
elif regularizer_name == "LSR":
    regularizer = LSR(
        nc=[32, 64, 128, 256],
        pretrained_denoiser=False,
    ).to(device)
elif regularizer_name == "LAR":
    regularizer = LocalAR(in_channels=1, pad=False, use_bias=True, n_patches=-1).to(
        device
    )
else:
    raise ValueError("Unknown model!")

if not os.path.isdir("weights"):
    os.mkdir("weights")
if not os.path.isdir(f"weights/adversarial_{problem}"):
    os.mkdir(f"weights/adversarial_{problem}")

# problem dependent parameters
physics, data_fidelity = get_operator(problem, device)
crop_size = 128 if regularizer_name == "LAR" else hyper_params.patch_size
transform = Compose(
    [
        RandomCrop(crop_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomApply([RandomRotation((90, 90))], p=0.5),
    ]
)

if problem == "Denoising":
    train_dataset = get_dataset("BSDS500_gray", test=False, transform=transform)
    val_dataset = get_dataset("BSDS500_gray", test=False, transform=CenterCrop(321))
    # splitting in training and validation set
    test_ratio = 0.1
    test_len = int(len(train_dataset) * 0.1)
    train_len = len(train_dataset) - test_len
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))

    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=hyper_params.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    fitting_set = torch.utils.data.Subset(val_dataset, range(0, 5))
    fitting_dataloader = torch.utils.data.DataLoader(
        fitting_set, batch_size=5, shuffle=True, drop_last=True
    )
elif problem == "CT":
    train_dataset = get_dataset("LoDoPaB", test=False)
    val_dataset = get_dataset("LoDoPaB", test=False)
    # splitting in training and validation set
    test_ratio = 0.003 if regularizer_name == "LAR" else 0.1
    test_len = int(len(train_dataset) * test_ratio)
    train_len = len(train_dataset) - test_len
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=hyper_params.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    fitting_set = get_dataset("LoDoPaB_val")
    # use smaller dataset for parameter fitting
    fitting_dataloader = torch.utils.data.DataLoader(
        fitting_set, batch_size=5, shuffle=True, drop_last=True, num_workers=8
    )

val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=True, drop_last=True
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"log_training_{regularizer_name}_{problem}"
    + "_AR_"
    + str(datetime.datetime.now())
    + ".log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

if only_fitting:
    ckp = torch.load(
        f"weights/adversarial_{problem}/{regularizer_name}_adversarial_for_{problem}.pt"
    )
    regularizer.load_state_dict(ckp)
else:
    regularizer = ar_training(
        regularizer,
        physics,
        data_fidelity,
        train_dataloader,
        val_dataloader,
        device=device,
        epochs=hyper_params.epochs,
        validation_epochs=hyper_params.val_epochs,
        lr=hyper_params.lr,
        lr_decay=hyper_params.lr_decay,
        mu=hyper_params.mu,
        LAR_eval=regularizer_name == "LAR",
        patch_size=hyper_params.patch_size if regularizer_name == "LAR" else None,
        dynamic_range_psnr=problem == "CT",
        patches_per_img=64 if regularizer_name == "LAR" else 8,
        logger=logger,
    )
    torch.save(
        regularizer.state_dict(),
        f"weights/adversarial_{problem}/{regularizer_name}_adversarial_for_{problem}.pt",
    )

if regularizer_name == "LAR":
    regularizer.pad = True
lmbd = estimate_lmbd(val_dataloader, physics, device)
lip = estimate_lip(regularizer, val_dataloader, device)
lmbd_est = lmbd / lip
print("Estimated Lambda for Fitting:", lmbd_est)

wrapped_regularizer = ParameterLearningWrapper(regularizer, device=device)
with torch.no_grad():
    wrapped_regularizer.alpha.copy_(torch.log(lmbd_est))

for p in wrapped_regularizer.parameters():
    p.requires_grad_(False)
    wrapped_regularizer.alpha.requires_grad_(True)
    wrapped_regularizer.scale.requires_grad_(True)

# parameter search on first five images of training set

bilevel_training(
    wrapped_regularizer,
    physics,
    data_fidelity,
    1,
    fitting_dataloader,
    val_dataloader,
    epochs=100,
    mode="IFT",
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    minres_tol=1e-4,
    lr=hyper_params.fitting_lr,
    lr_decay=0.98 if problem == "CT" else 0.99,
    reg=False if problem == "CT" else True,
    device=device,
    verbose=True,
    validation_epochs=10,
    dynamic_range_psnr=True if problem == "CT" else False,
    adabelief=True,
    logger=logger,
)

torch.save(
    wrapped_regularizer.state_dict(),
    f"weights/adversarial_{problem}/{regularizer_name}_adversarial_for_{problem}_fitted.pt",
)

print("Final alpha: ", torch.exp(wrapped_regularizer.alpha))
print("Final scale: ", wrapped_regularizer.scale)
