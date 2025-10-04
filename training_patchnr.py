"""
Training the PatchNR prior. See the readme file (section "Reproduce the Training Runs (Experiment 1 and 3)") 
for details.
"""

import torch
from torch.utils.data import DataLoader
from deepinv.datasets import PatchDataset

from tqdm import tqdm
import numpy as np
import os
import argparse
from dataset import get_dataset
from torchvision.transforms import RandomCrop, CenterCrop
from evaluation import evaluate

from priors import ParameterLearningWrapper, PatchNR
from operators import get_operator
from training_methods import bilevel_training
import logging
import datetime


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument(
    "--problem", type=str, default="Denoising", choices=["Denoising", "CT"]
)
parser.add_argument("--only_fitting", type=bool, default=False)

inp = parser.parse_args()

problem = inp.problem
only_fitting = inp.only_fitting
print("only fitting: ", only_fitting)
if not os.path.isdir("weights"):
    os.mkdir("weights")
if not os.path.isdir("weights/patchnr"):
    os.mkdir("weights/patchnr")

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="log_training_"
    + problem
    + "_PatchNR_"
    + str(datetime.datetime.now())
    + ".log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

# problem dependent parameters
if problem == "Denoising":
    physics, data_fidelity = get_operator(problem, device)
    dataset = get_dataset("BSDS500_gray", test=False, transform=RandomCrop(128))
    train_on = "BSD500"

    val_dataset = get_dataset("BSDS500_gray", test=False)
    # splitting in training and validation set
    fitting_set = torch.utils.data.Subset(val_dataset, range(0, 5))
    fitting_dataloader = torch.utils.data.DataLoader(
        fitting_set, batch_size=1, shuffle=True, drop_last=True
    )
    test_ratio = 0.1
    test_len = int(len(dataset) * 0.1)
    train_len = len(dataset) - test_len
    train_set = torch.utils.data.Subset(dataset, range(train_len))
    val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(dataset)))

    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=True, drop_last=True
    )
    min_lmbd = 12.0
    max_lmbd = 20.0
    patchnr_epochs = 20
    patchnr_batch_size = 1024
elif problem == "CT":
    dataset = get_dataset("LoDoPaB", test=False, transform=RandomCrop(128))
    physics, data_fidelity = get_operator(problem, device)
    train_on = "LoDoPab"

    val_dataset = get_dataset("LoDoPaB", test=False)
    # splitting in training and validation set
    fitting_set = torch.utils.data.Subset(val_dataset, range(0, 5))
    fitting_dataloader = torch.utils.data.DataLoader(
        fitting_set, batch_size=1, shuffle=True, drop_last=True
    )
    test_ratio = 0.003
    test_len = int(len(dataset) * 0.1)
    train_len = len(dataset) - test_len
    train_set = torch.utils.data.Subset(dataset, range(train_len))
    val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(dataset)))

    val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=True, drop_last=True
    )
    min_lmbd = 280.0
    max_lmbd = 320.0
    patchnr_epochs = 4
    patchnr_batch_size = 2048
else:
    raise NotImplementedError

patch_size = 6
patchnr_subnetsize = 512
n_patches = 10000  # -1 for all patches
regularizer = PatchNR(
    patch_size=patch_size,
    channels=1,
    num_layers=5,
    sub_net_size=patchnr_subnetsize,
    device=device,
    n_patches=n_patches,
    pad=True,
)

if only_fitting:
    ckp = torch.load(f"weights/patchnr/patchnr_{patch_size}x{patch_size}_{train_on}.pt")
    regularizer.load_state_dict(ckp)
else:
    train_imgs = []
    for i in range(len(dataset)):
        train_imgs.append(dataset[i].unsqueeze(0).float())

    train_imgs = torch.concat(train_imgs)

    verbose = True
    train_dataset = PatchDataset(train_imgs, patch_size=patch_size)

    patchnr_learning_rate = 5e-4

    patchnr_dataloader = DataLoader(
        train_dataset,
        batch_size=patchnr_batch_size,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(
        regularizer.normalizing_flow.parameters(), lr=patchnr_learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=patchnr_epochs, eta_min=patchnr_learning_rate / 100.0
    )
    print("Start training PatchNR")
    logger.info("Start training PatchNR")
    for epoch in range(patchnr_epochs):
        mean_loss = []
        with tqdm(total=len(patchnr_dataloader)) as pbar:
            for idx, batch in enumerate(patchnr_dataloader):
                optimizer.zero_grad()

                x = batch.to(device)
                x = x + 1 / 256.0 * torch.rand_like(x)  # add small dequantisation noise
                latent_x, logdet = regularizer.normalizing_flow(
                    x
                )  # x -> z (we never need the other direction)

                # Compute the Kullback Leibler loss
                logpz = 0.5 * torch.sum(latent_x**2, -1)

                nll = logpz - logdet

                loss_total = nll.mean()
                mean_loss.append(loss_total.item())

                loss_total.backward()  # Backward the total loss
                optimizer.step()  # Optimizer step

                pbar.update(1)
                pbar.set_description(f"Loss {np.round(loss_total.item(), 5)}")

        log_string = f"[Epoch {epoch+1} / {patchnr_epochs}] Train Loss: {np.mean(mean_loss):.2E} Step Size: {scheduler.get_last_lr()[0]:.3E}"
        print(log_string)
        logger.info(log_string)

        scheduler.step()
        torch.save(
            regularizer.state_dict(),
            f"weights/patchnr/patchnr_{patch_size}x{patch_size}_{train_on}.pt",
        )

# bilevel learning does not find into memory, use a line search instead

best_mean_psnr = -float("inf")

lambdas = np.linspace(min_lmbd, max_lmbd, 25)
for lamb in lambdas:
    mean_psnr, x_out, y_out, recon_out = evaluate(
        physics=physics,
        data_fidelity=data_fidelity,
        dataset=fitting_set,
        regularizer=regularizer,
        lmbd=lamb,
        step_size=1e-3,
        max_iter=1000,
        tol=1e-4,
        adam=True,
        only_first=False,
        device=device,
        verbose=False,
        adaptive_range=True if problem == "CT" else False,
    )
    print(f"Mean PSNR for lambda {lamb}: {mean_psnr:.2f}")
    logger.info(f"Mean PSNR for lambda {lamb}: {mean_psnr:.2f}")
    if mean_psnr > best_mean_psnr:
        best_mean_psnr = mean_psnr
        best_lamb = lamb

print(f"Best lambda: {best_lamb}")
logger.info(f"Best lambda: {best_lamb}")


data = {
    "patch_size": patch_size,
    "n_patches": n_patches,
    "patchnr_subnetsize": patchnr_subnetsize,
    "weights": regularizer.state_dict(),
    "lambda": best_lamb,
}
torch.save(
    data, f"weights/patchnr/patchnr_{patch_size}x{patch_size}_{train_on}_fitted.pt"
)
