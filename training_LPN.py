"""Train LPN by prox matching for denoising on BSD or LoDoPaB."""

import argparse
import os

import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomApply,
    CenterCrop,
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
)

from dataset import get_dataset
from priors.lpn.lpn import LPNPrior
from training_methods.lpn_training import Validator, lpn_training

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="BSD",
    choices=["BSD", "LoDoPaB"],
    help="Dataset to use for training LPN",
)
parser.add_argument(
    "--noise_level", type=float, default=0.1, help="Noise level for training"
)
parser.add_argument("--ckpt_dir", type=str, default=None)
args = parser.parse_args()

###############################################################################
# Select default parameters
if args.dataset == "BSD":
    ckpt_dir = args.ckpt_dir or "weights/lpn_64_bsd"
    noise_level = args.noise_level or 0.1
    batch_size = 64
elif args.dataset == "LoDoPaB":
    ckpt_dir = args.ckpt_dir or "weights/lpn_64_ct"
    noise_level = args.noise_level or 0.1
    batch_size = 128
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

crop_size = 64
physics = None
data_fidelity = None
###############################################################################


###############################################################################
# Create dataset and dataloaders
transform = Compose(
    [
        CenterCrop(crop_size),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomApply([RandomRotation((90, 90))], p=0.5),
    ]
)

if args.dataset == "BSD":
    train_dataset = get_dataset("BSDS500_gray", test=False, transform=transform)
    val_dataset = get_dataset(
        "BSDS500_gray",
        test=False,
    )
    # split train and val
    test_ratio = 0.1
    test_len = int(len(train_dataset) * test_ratio)
    train_len = len(train_dataset) - test_len
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))

    # val_set = get_dataset("BSD68", transform=CenterCrop(crop_size))  # test on BSD68

elif args.dataset == "LoDoPaB":

    train_dataset = get_dataset("LoDoPaB", test=False, transform=transform)
    val_dataset = get_dataset("LoDoPaB", test=False)
    # split train and val
    val_ratio = 0.1
    val_len = int(len(train_dataset) * val_ratio)
    train_len = len(train_dataset) - val_len
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))


# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True, num_workers=8
)
data = next(iter(train_dataloader))
print(data.shape, data.min(), data.max())

###############################################################################


###############################################################################
# Training
###############################################################################
# define regularizer
regularizer = LPNPrior().to(device)

os.makedirs(ckpt_dir, exist_ok=True)
regularizer = lpn_training(
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
torch.save(regularizer.state_dict(), f"{ckpt_dir}/LPN.pt")
