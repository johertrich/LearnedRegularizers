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

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

dataset = get_dataset("BSDS500_gray", test=False, transform=CenterCrop(300)) #RandomCrop(256))

print("Length of BSD Dataset: ", len(dataset))

train_imgs = [] 
for i in range(100):
    train_imgs.append(dataset[i].unsqueeze(0).float())

train_imgs = torch.concat(train_imgs)

print(train_imgs.shape)

patch_size = 6
verbose = True
train_dataset = PatchDataset(train_imgs, patch_size=patch_size, transforms=None)

patchnr_subnetsize = 512
patchnr_epochs = 10
patchnr_batch_size = 512
patchnr_learning_rate = 1e-4

patchnr_dataloader = DataLoader(
    train_dataset,
    batch_size=patchnr_batch_size,
    shuffle=True,
    drop_last=True,
)

patch_nr = PatchNR(patch_size=patch_size, channels=1,num_layers=5, sub_net_size=patchnr_subnetsize, device=device, n_patches=-1)

optimizer = torch.optim.Adam(
        patch_nr.normalizing_flow.parameters(), lr=patchnr_learning_rate
    )

pretrain = None #"patchnr.pt"

if pretrain is None:
    for epoch in range(patchnr_epochs):
        mean_loss = [] 
        with tqdm(total=len(patchnr_dataloader)) as pbar:
            for idx, batch in enumerate(patchnr_dataloader):
                optimizer.zero_grad()

                x = batch[0].to(device)

                invs, jac_inv = patch_nr.normalizing_flow(x) # x -> z (we never need the other direction)

                # Compute the Kullback Leibler loss
                loss_total = torch.mean(
                    0.5 * torch.sum(invs.view(invs.shape[0], -1) ** 2, -1)
                    - jac_inv.view(invs.shape[0])
                )
                mean_loss.append(loss_total.item())

                loss_total.backward()  # Backward the total loss
                optimizer.step()  # Optimizer step
                
                pbar.update(1)
                pbar.set_description(f"Loss {np.round(loss_total.item(), 5)}")

        print("Mean loss: ", np.mean(mean_loss))

    torch.save(patch_nr.normalizing_flow.state_dict(), "patchnr.pt")
else:
    patch_nr.normalizing_flow.load_state_dict(torch.load(pretrain))

image = dataset[10].unsqueeze(0).float().to(device)


sigma = 0.01 #0.1
noise_model = GaussianNoise(sigma)
physics = Downsampling(img_size=(1,256,256), factor=2, device=device, noise_model=noise_model) #Denoising(device=device, noise_model=noise_model)
data_fidelity = L2()

observation = physics(image)

print(observation)

print(image.shape, observation.shape)

optim_steps = 400  #1000
lr_variational_problem = 0.01

for n_patches in [10, 100, 1000, 5000, 10000]:
    patch_nr.n_patches = n_patches
    print(f"n_patches {patch_nr.n_patches} value of prior {patch_nr.g(image).item()} (second evaluation {patch_nr.g(image).item()})")


def minimize_variational_problem(prior, lam, optim_steps):
    imgs = torch.zeros_like(image)
    #imgs = physics.A_adjoint(observation).detach().clone()
    imgs.requires_grad_(True)
    optimizer = torch.optim.SGD([imgs], lr=lr_variational_problem)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optim_steps, eta_min=lr_variational_problem/10.0)
    for i in (progress_bar := tqdm(range(optim_steps))):
        optimizer.zero_grad()
        loss = data_fidelity(imgs, observation, physics).mean() + lam * prior.g(imgs)
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_description("Step {} || Loss {} || {}".format(i + 1, loss.item(), PSNR()(imgs, image).item()))
    return imgs.detach()

lam = 2.0  #6.4

total_patch_number = (image.shape[2] - patch_size + 1) * (image.shape[3] - patch_size + 1)
print("total patch number: ", total_patch_number)

patch_nr.n_patches = -1 


recon_patchnr = minimize_variational_problem(patch_nr, lam, optim_steps)

#patch_nr.n_patches = 1000 

#recon_patchnr_subset = minimize_variational_problem(patch_nr, lam, optim_steps*10)

psnr_patchnr = PSNR()(recon_patchnr, image).cpu().squeeze().numpy()
#psnr_patchnr_subset = PSNR()(recon_patchnr_subset, image).cpu().squeeze().numpy()

#print(psnr_patchnr, psnr_patchnr_subset)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(16,7))

im = ax1.imshow(image[0,0].cpu().numpy(), cmap="gray")
fig.colorbar(im, ax=ax1)
im = ax2.imshow(observation[0,0].cpu().numpy(), cmap="gray")
fig.colorbar(im, ax=ax2)

im = ax3.imshow(recon_patchnr[0,0].cpu().numpy(), cmap="gray")
fig.colorbar(im, ax=ax3)

ax3.set_title(f"PatchNR, all patches: {psnr_patchnr}")
im = ax4.imshow(physics.A_adjoint(observation)[0,0].detach().cpu().numpy(), cmap="gray")
fig.colorbar(im, ax=ax4)

#ax4.imshow(recon_patchnr_subset[0,0].cpu().numpy(), cmap="gray")
ax4.set_title("A adjoint")
#ax4.set_title(f"PatchNR, subset patches: {psnr_patchnr_subset}")
plt.savefig("patchnr.png")
#plt.show()
plt.close()   
