# -*- coding: utf-8 -*-
"""
Training of post-processing UNet 

@author: Alexander
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import Dataset
from dataset import get_dataset
from tqdm import tqdm
import numpy as np
import yaml

from operators import get_operator

import deepinv as dinv

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


class FBPDataset(Dataset):
    def __init__(self, dataset, physics, device):
        self.dataset = dataset
        self.physics = physics
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx].unsqueeze(0).to(self.device)
        y = self.physics(x)

        return x.squeeze(0), y.squeeze(0)


dataset = FBPDataset(dataset, physics, device)
x, y = dataset[0]

test_ratio = 0.01

# splitting in training and validation set
test_len = int(len(dataset) * test_ratio)
train_len = len(dataset) - test_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])

batch_size = 16
shuffle = True

train_dl = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=shuffle
)
val_dl = torch.utils.data.DataLoader(val_set, batch_size=1)

model = dinv.models.ArtifactRemoval(
    dinv.models.UNet(1, 1, scales=5, batch_norm=True).to(device), mode="pinv"
)

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    train_dataloader=train_dl,
    eval_dataloader=val_dl,
    epochs=100,
    losses=dinv.loss.SupLoss(metric=dinv.metric.MSE()),
    metrics=dinv.metric.PSNR(),
    device=device,
    plot_images=False,
    show_progress_bar=True,
    save_path="supervised_training/fbpunet",
    plot_measurements=False,
    ckp_interval=10,
)

_ = trainer.train()
