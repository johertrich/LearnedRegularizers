"""Train LPN by prox matching."""

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

problem = "Denoising"
crop_size = 64

# problem dependent parameters
if problem == "Denoising":
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
    train_dataset = get_dataset("BSDS500_gray", test=False, transform=transform)
    test_transform = None  # CenterCrop(crop_size)
    val_dataset = get_dataset(
        "BSDS500_gray",
        test=False,
        transform=test_transform,
    )
    # split train and val
    test_ratio = 0.1
    test_len = int(len(train_dataset) * test_ratio)
    train_len = len(train_dataset) - test_len
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))

    # val_set = get_dataset("BSD68", transform=test_transform)  # test on BSD68
    # val_set = torch.utils.data.Subset(
    #     get_dataset("BSDS500_gray", test=False, transform=test_transform), range(68)
    # )  # test on train


# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True, drop_last=True, num_workers=8
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

ckpt_dir = "weights/lpn_64_neg1_pm"
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
    num_steps=320000,
    num_steps_pretrain=160000,
    validate_every_n_steps=10000,
    ckpt_dir=ckpt_dir,
    loss_type="pm",
)
torch.save(regularizer.state_dict(), f"{ckpt_dir}/simple_LPN.pt")

###############################################################################
# Evaluate the model
###############################################################################
# regularizer = LPNPrior(model_name="lpn_64_neg1").to(device)
# ckpt_dir = "weights/lpn_64_neg1_pm"
# regularizer.load_state_dict(torch.load(f"{ckpt_dir}/simple_LPN.pt"))
# validator = Validator(val_dataloader, writer=None, sigma_noise=25.5 / 255.0)
# validator.validate(regularizer.lpn)
# for i, img_dict in enumerate(validator.img_list):
#     # save the images in a grid
#     fig, axs = plt.subplots(1, 3, figsize=(10, 5), dpi=300)
#     for j, key in enumerate(img_dict.keys()):
#         ax = axs[j]
#         img = img_dict[key]
#         img = img[0].cpu().numpy().transpose(1, 2, 0)
#         img = img.clip(0, 1)
#         img = (img * 255).astype(np.uint8).squeeze()
#         ax.imshow(img, cmap="gray", vmin=0, vmax=255)
#         ax.set_title(key)
#         ax.axis("off")
#         if key == "pred":
#             ax.set_title(
#                 f"{key}, PSNR: {validator.psnr_list[i]:.2f}, SSIM: {validator.ssim_list[i]:.4f}"
#             )
#     fig.tight_layout()
#     os.makedirs(os.path.join(ckpt_dir, "results", "lpn"), exist_ok=True)
#     fig.savefig(
#         os.path.join(ckpt_dir, "results", "lpn", f"{i}.png"), bbox_inches="tight"
#     )
#     plt.close(fig)
