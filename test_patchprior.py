import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import numpy as np 
from torch.distributions.multivariate_normal import MultivariateNormal

#from deepinv.optim import EPLL
from deepinv.physics import GaussianNoise, Denoising, Inpainting
from deepinv.loss.metric import PSNR
from deepinv.utils import plot
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.datasets import PatchDataset
from deepinv.optim.data_fidelity import L2
from tqdm import tqdm
from torchvision.utils import make_grid

from dataset import get_dataset
from torchvision.transforms import RandomCrop
from priors.patchnr import PatchNR
from priors.epll import EPLL


device = "cuda" if torch.cuda.is_available() else "cpu"


dataset = get_dataset("BSDS500_gray", test=False, transform=RandomCrop(256))

image = dataset[15].unsqueeze(0).float().to(device)

patch_size = 8
model_epll = EPLL(channels=image.shape[1], patch_size=patch_size, device=device, pretrained=None)
model_epll.GMM.load_state_dict(torch.load("gmm.pt"))

means = model_epll.GMM.mu.data
#means = means.view(model_epll.GMM.n_components, patch_size, patch_size).unsqueeze(1)
print(model_epll.GMM._cov.shape)

#print("Weights:")
#print(model_epll.GMM._weights)
"""
for i in range(200):
    normal = MultivariateNormal(means[i,:], covariance_matrix=model_epll.GMM._cov[i,:,:])

    sample = normal.sample([25])

    fig, axes = plt.subplots(5,5)

    for idx, ax in enumerate(axes.ravel()):

        ax.axis("off")
        ax.imshow(sample[idx,:].view(patch_size, patch_size).cpu().numpy(), cmap="gray")

    plt.show()
"""
#print("sample: ", sample.shape)
print(means.shape)

#grid_img = make_grid(means.detach().cpu(), nrow=10, normalize=True, padding=2)
#grid_np = grid_img.permute(1, 2, 0).numpy()

#plt.figure(figsize=(17, 9))
#plt.imshow(np.clip(grid_np, 0, 1))
#plt.axis("off")
#plt.title("Learned Means")
#plt.show()


patch_nr = PatchNR(patch_size=patch_size, channels=1,num_layers=5, sub_net_size=512, device=device, n_patches=5000)#-1)
#patch_nr.normalizing_flow.load_state_dict(torch.load("patchnr.pt"))


sigma = 0.1
noise_model = GaussianNoise(sigma)
physics = Denoising(device=device, noise_model=noise_model)
data_fidelity = L2()

observation = physics(image)

with torch.no_grad():
    recon_epll = model_epll(observation, physics, batch_size=-1)

optim_steps = 600
lr_variational_problem = 0.005

def minimize_variational_problem(prior, lam):
    imgs = observation.detach().clone() * 0
    imgs.requires_grad_(True)
    optimizer = torch.optim.Adam([imgs], lr=lr_variational_problem)
    for i in (progress_bar := tqdm(range(optim_steps))):
        optimizer.zero_grad()
        loss = data_fidelity(imgs, observation, physics).mean() + lam * prior.g(imgs)
        loss.backward()
        optimizer.step()
        progress_bar.set_description("Step {} || Loss {}".format(i + 1, loss.item()))
    return imgs.detach()

recon_patchnr = minimize_variational_problem(patch_nr, 1.75)
recon_epll_var = minimize_variational_problem(model_epll, 1.5)


psnr_patchnr = PSNR()(recon_patchnr, image).cpu().numpy()
psnr_epll = PSNR()(recon_epll, image).cpu().numpy()
psnr_epll2 = PSNR()(recon_epll_var, image).cpu().numpy()

print(psnr_patchnr, psnr_epll)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)

ax1.imshow(image[0,0].cpu().numpy(), cmap="gray")
ax1.set_title("Ground truth")
ax2.imshow(observation[0,0].cpu().numpy(), cmap="gray")
ax2.set_title("Observation")
ax3.imshow(recon_epll[0,0].cpu().numpy(), cmap="gray")
ax3.set_title(f"Reco EPLL, PNSR {psnr_epll}")
ax4.imshow(recon_epll_var[0,0].cpu().numpy(), cmap="gray")
ax4.set_title(f"Reco EPLL (variational), PNSR {psnr_epll2}")
ax5.imshow(recon_patchnr[0,0].cpu().numpy(), cmap="gray")
ax5.set_title(f"Reco PatchNR, PNSR {psnr_patchnr}")

plt.show()
