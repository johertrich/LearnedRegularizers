"""
Evaluation Script for the FBPUnet

@author: Alex
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from operators import get_evaluation_setting

import torch
from deepinv.loss.metric import PSNR

import deepinv as dinv
from torch.utils.data import Dataset

problem = "CT"

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device: ", device)
torch.random.manual_seed(0)  # make results deterministic

dataset, physics, data_fidelity = get_evaluation_setting(problem, device)


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


fbp_dataset = FBPDataset(dataset, physics, device)

fbp_dataloader = torch.utils.data.DataLoader(fbp_dataset, batch_size=1, shuffle=False)


model = dinv.models.ArtifactRemoval(
    dinv.models.UNet(1, 1, scales=5, batch_norm=True).to(device), mode="pinv"
)

print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

model.load_state_dict(
    torch.load("supervised_training/fbpunet/ckp_best.pth.tar", map_location=device)[
        "state_dict"
    ]
)
model.eval()
model.to(device)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
psnr = PSNR(max_pixel=None)

dinv.test(
    model, fbp_dataloader, physics, metrics=psnr, show_progress_bar=True, device=device
)

