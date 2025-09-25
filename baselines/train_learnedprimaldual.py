# -*- coding: utf-8 -*-
"""
Training of LearnedPrimalDualNet

@author: Alexander
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from dataset import get_dataset
from tqdm import tqdm
import numpy as np
import yaml
from deepinv.loss.metric import PSNR

from operators import get_operator
from network.learned_primal_dual import PrimalDualNet

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "CT"
dataset = get_dataset("LoDoPaB", test=False, transform=None)
physics, data_fidelity = get_operator(problem, device)
lmbd = 1.0
test_ratio = 0.01

# splitting in training and validation set
test_len = int(len(dataset) * test_ratio)
train_len = len(dataset) - test_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])

batch_size = 16
shuffle = True

cfg = {
    "lr": 1e-4,
    "num_epochs": 1000,
    "model_params": {
        "n_iter": 6,
        "n_primal": 4,
        "n_dual": 4,
        "use_sigmoid": False,
        "n_layer": 4,
        "internal_ch": 32,
        "kernel_size": 3,
        "batch_norm": True,
        "lrelu_coeff": 0.2,
    },
}


model = PrimalDualNet(
    image_width=362,
    num_angles=60,
    n_iter=cfg["model_params"]["n_iter"],
    op=physics.A,
    op_adj=physics.A_dagger,
    op_init=physics.A_dagger,
    n_primal=cfg["model_params"]["n_primal"],
    n_dual=cfg["model_params"]["n_dual"],
    use_sigmoid=cfg["model_params"]["use_sigmoid"],
    n_layer=cfg["model_params"]["n_layer"],
    internal_ch=cfg["model_params"]["internal_ch"],
    kernel_size=cfg["model_params"]["kernel_size"],
    batch_norm=cfg["model_params"]["batch_norm"],
    lrelu_coeff=cfg["model_params"]["lrelu_coeff"],
)
model.train()
model.to(device)

train_dl = torch.utils.data.DataLoader(train_set, batch_size=8)
val_dl = torch.utils.data.DataLoader(val_set, batch_size=1)

print("Length of training set: ", len(train_set))
print("Length of val set: ", len(val_set))

optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=cfg["num_epochs"], eta_min=cfg["lr"] / 100.0
)

min_loss = 100

psnr = PSNR(max_pixel=None)

for epoch in range(cfg["num_epochs"]):

    model.train()
    for idx, batch in (progress_bar := tqdm(enumerate(train_dl), total=len(train_dl))):
        optimiser.zero_grad()
        x = batch.to(device)
        y = physics(x)
        x_pred = model(y)
        loss = torch.mean((x_pred - x) ** 2)
        loss.backward()

        optimiser.step()

        progress_bar.set_description(
            "Epoch {} || Step {} || Loss {:.7f} ".format(
                epoch + 1, idx + 1, loss.item()
            )
        )
    scheduler.step()
    model.eval()
    with torch.no_grad():
        mean_loss = []
        val_psnr_epoch = 0
        for batch in val_dl:
            x = batch.to(device)
            y = physics(x)

            x_pred = model(y)

            val_psnr_epoch += psnr(x_pred, x).mean().item()

            loss = torch.mean((x_pred - x) ** 2)

            mean_loss.append(loss.item())

    mean_val_psnr = val_psnr_epoch / len(val_dl)

    print("Validation Loss: ", np.mean(mean_loss), " , PSNR = ", mean_val_psnr)

    if np.mean(mean_loss) < min_loss:
        min_loss = np.mean(mean_loss)
        print("new min loss: save model")
        with open("suoervised_training/lpd_config.yaml", "w") as f:
            yaml.dump(cfg, f)
        torch.save(model.state_dict(), "suoervised_training/lpdnet.pt")
