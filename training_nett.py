"""
Training NETT. See the readme file (section "Reproduce the Training Runs (Experiment 1 and 3)") 
for details.
"""

from priors import NETT, ParameterLearningWrapper, LSR
import torch
from deepinv.physics import Denoising, GaussianNoise, Tomography
from training_methods import NETT_training, bilevel_training
from deepinv.optim import L2, L1
from dataset import get_dataset
from operators import get_operator
from torchvision.transforms import (
    RandomCrop,
    RandomAutocontrast,
    CenterCrop,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
)
from torch.utils.data import Subset as subset
from torchvision import transforms
import logging
import datetime
import numpy as np
import argparse

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument("--problem", type=str, default="Denoising")
inp = parser.parse_args()

problem = inp.problem
algorithm = "Adam"  # or "MAID", "Adam", "AdamW", "ISGD_Momentum"

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="log_training_NETT_"
    + str(problem)
    + "_"
    + str(datetime.datetime.now())
    + ".log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

flip_transform = transforms.Compose([RandomHorizontalFlip(), RandomVerticalFlip()])

if problem == "Denoising":
    train_dataset = get_dataset(
        "BSDS500_gray", test=False, transform=flip_transform, rotate=True
    )
    val_dataset = get_dataset("BSDS500_gray", test=False)
    # splitting in training and validation set
    test_ratio = 0.1
    test_len = int(len(train_dataset) * 0.1)
    train_len = len(train_dataset) - test_len
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))
    fit_set = torch.utils.data.Subset(val_dataset, range(0, 5))
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=4, shuffle=True, drop_last=True, num_workers=8
    )
    lmbd = 5
    num_epochs = 200
    validation_epochs = 10
    fit_set = torch.utils.data.Subset(
        get_dataset("BSDS500_gray", test=False, rotate=True), range(5)
    )
elif problem == "CT":
    train_dataset = get_dataset("LoDoPaB", test=False, transform=flip_transform)
    val_dataset = get_dataset("LoDoPaB", test=False)
    # splitting in training and validation set
    test_ratio = 0.1
    test_len = int(len(train_dataset) * 0.1)
    train_len = len(train_dataset) - test_len
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    val_set = torch.utils.data.Subset(
        val_dataset, range(len(train_dataset) - 40, len(train_dataset))
    )
    fit_set = get_dataset("LoDoPaB_val")
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=4, shuffle=True, drop_last=True, num_workers=8
    )
    lmbd = 600
    num_epochs = 60
    validation_epochs = 5

val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True, num_workers=8
)
physics, data_fidelity = get_operator(problem, device=device)


# define regularizer
regularizer = NETT(
    in_channels=1, out_channels=1, hidden_channels=64, padding_mode="zeros"
).to(device)

regularizer = NETT_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    device=device,
    lr=1e-4,  # 1e-4
    num_epochs=num_epochs,
    lr_decay=0.1 ** (1 / num_epochs),
    save_best=True,
    logger=logger,
    p=0.5,
    weight_dir="weights/NETT_" + problem + ".pt",
    dynamic_range_psnr=problem == "CT",
    validation_epochs=validation_epochs,
)

wrapped_regularizer = ParameterLearningWrapper(regularizer, device=device)
wrapped_regularizer.alpha.data += np.log(lmbd)

torch.save(wrapped_regularizer.state_dict(), f"weights/NETT_{problem}.pt")

for p in wrapped_regularizer.parameters():
    p.requires_grad_(False)
wrapped_regularizer.alpha.requires_grad_(True)

fit_dataloader = torch.utils.data.DataLoader(
    fit_set, batch_size=5, shuffle=True, drop_last=True
)

bilevel_training(
    wrapped_regularizer,
    physics,
    data_fidelity,
    1,
    fit_dataloader,
    val_dataloader,
    epochs=100,
    mode="JFB",
    lower_level_step_size=1e-1,
    lower_level_max_iter=1000,
    lower_level_tol_train=1e-4,
    lower_level_tol_val=1e-4,
    lr=0.01,
    lr_decay=0.98,
    device=device,
    verbose=False,
    validation_epochs=10,
    dynamic_range_psnr=problem == "CT",
    adabelief=True,
    reg=False,
    logger=logger,
)

torch.save(wrapped_regularizer.state_dict(), f"weights/NETT_{problem}_fitted.pt")
