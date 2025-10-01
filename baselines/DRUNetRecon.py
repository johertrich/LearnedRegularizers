# evaluates the DRUNet for Denoising

from deepinv.models import DRUNet
from operators import get_evaluation_setting
from deepinv.loss.metric import PSNR
import torch
import numpy as np

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

torch.manual_seed(0)
psnr = PSNR()
model = DRUNet(in_channels=1, out_channels=1).to(device)
dataset, physics, data_fidelity = get_evaluation_setting("Denoising", device)
psnrs = []
for i in range(len(dataset)):
    x = dataset[i].unsqueeze(0).to(device)
    y = physics(x)
    recon = model(y, sigma=0.1)
    psnrs.append(psnr(recon, x).item())
print("Mean PSNR DRUNet for Denoising:", np.mean(psnrs))
