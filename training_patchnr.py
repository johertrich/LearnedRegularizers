import torch
from torch.utils.data import DataLoader
from deepinv.datasets import PatchDataset
from deepinv import Trainer
from deepinv.physics import Denoising, UniformNoise, GaussianNoise, Downsampling
from deepinv.loss.metric import PSNR
from deepinv.utils import plot
from tqdm import tqdm
from deepinv.optim.data_fidelity import L2
import numpy as np

from dataset import get_dataset
from torchvision.transforms import RandomCrop, CenterCrop

from priors.patchnr import PatchNR

import matplotlib.pyplot as plt
from operators import get_operator

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

problem = "Denoising" # "Denoising" "CT"

# problem dependent parameters
if problem == "Denoising":
    physics , data_fidelity = get_operator(problem, device)
    dataset = get_dataset("BSDS500_gray", test=False, transform=RandomCrop(128))
    train_on = "BSD500"
elif problem == "CT":
    dataset = get_dataset("LoDoPaB", test=False, transform=RandomCrop(128))
    physics , data_fidelity = get_operator(problem, device)
    train_on = "LoDoPab"
else:
    raise NotImplementedError

print("Length of Dataset: ", len(dataset))

train_imgs = []
for i in range(len(dataset)):
    train_imgs.append(dataset[i].unsqueeze(0).float())

train_imgs = torch.concat(train_imgs)

print(train_imgs.shape)

patch_size = 6
verbose = True
train_dataset = PatchDataset(train_imgs, patch_size=patch_size, transforms=None)

patchnr_subnetsize = 512
patchnr_epochs = 20
patchnr_batch_size = 1024
patchnr_learning_rate = 5e-4

patchnr_dataloader = DataLoader(
    train_dataset,
    batch_size=patchnr_batch_size,
    shuffle=True,
    drop_last=True,
)

patch_nr = PatchNR(
    patch_size=patch_size,
    channels=1,
    num_layers=5,
    sub_net_size=patchnr_subnetsize,
    device=device,
    n_patches=-1,
)

optimizer = torch.optim.Adam(
    patch_nr.normalizing_flow.parameters(), lr=patchnr_learning_rate
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=patchnr_epochs, eta_min=patchnr_learning_rate/100.)

for epoch in range(patchnr_epochs):
    mean_loss = []
    with tqdm(total=len(patchnr_dataloader)) as pbar:
        for idx, batch in enumerate(patchnr_dataloader):
            optimizer.zero_grad()

            x = batch[0].to(device)
            x = x + 1/256. * torch.rand_like(x) # add small dequantisation noise
            latent_x, logdet = patch_nr.normalizing_flow(x)  # x -> z (we never need the other direction)

            # Compute the Kullback Leibler loss
            logpz = 0.5 * torch.sum(latent_x ** 2, -1)
            
            nll = logpz - logdet

            loss_total = nll.mean()
            mean_loss.append(loss_total.item())

            loss_total.backward()  # Backward the total loss
            optimizer.step()  # Optimizer step

            pbar.update(1)
            pbar.set_description(f"Loss {np.round(loss_total.item(), 5)}")

    print(f"[Epoch {epoch+1} / {patchnr_epochs}] Train Loss: {np.mean(mean_loss):.2E} Step Size: {scheduler.get_last_lr()[0]:.3E}")

    scheduler.step()
    torch.save(patch_nr.normalizing_flow.state_dict(), f"weights/patchnr_{patch_size}x{patch_size}_{train_on}.pt")
