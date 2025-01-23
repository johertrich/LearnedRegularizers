import torch
from torch.utils.data import DataLoader
from deepinv.datasets import PatchDataset
from deepinv import Trainer
from deepinv.physics import Denoising, UniformNoise, GaussianNoise
from deepinv.loss.metric import PSNR
from deepinv.utils import plot
from tqdm import tqdm
from deepinv.optim.data_fidelity import L2

from dataset import get_dataset
from torchvision.transforms import RandomCrop

from priors.patchnr import PatchNR

import matplotlib.pyplot as plt 

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

dataset = get_dataset("BSDS500_gray", test=False, transform=RandomCrop(128))

train_imgs = [] 
for i in range(10):
    train_imgs.append(dataset[i].unsqueeze(0).float())

train_imgs = torch.concat(train_imgs)

print(train_imgs.shape)

patch_size = 6
verbose = True
train_dataset = PatchDataset(train_imgs, patch_size=patch_size, transforms=None)

patchnr_subnetsize = 128
patchnr_epochs = 6
patchnr_batch_size = 32
patchnr_learning_rate = 1e-4

patchnr_dataloader = DataLoader(
    train_dataset,
    batch_size=patchnr_batch_size,
    shuffle=True,
    drop_last=True,
)

class NFTrainer(Trainer):
    def compute_loss(self, physics, x, y, train=True):
        logs = {}

        self.optimizer.zero_grad()  # Zero the gradients

        # Evaluate reconstruction network
        invs, jac_inv = self.model(y)

        # Compute the Kullback Leibler loss
        loss_total = torch.mean(
            0.5 * torch.sum(invs.view(invs.shape[0], -1) ** 2, -1)
            - jac_inv.view(invs.shape[0])
        )
        current_log = (
            self.logs_total_loss_train if train else self.logs_total_loss_eval
        )
        current_log.update(loss_total.item())
        logs["TotalLoss"] = current_log.avg

        if train:
            loss_total.backward()  # Backward the total loss
            self.optimizer.step()  # Optimizer step

        return invs, logs
    

patch_nr = PatchNR(patch_size=patch_size, channels=1,num_layers=10, sub_net_size=patchnr_subnetsize, device=device, n_patches=-1)


optimizer = torch.optim.Adam(
        patch_nr.normalizing_flow.parameters(), lr=patchnr_learning_rate
    )

pretrain = "patchnr.pt"

if pretrain is None:

    trainer = NFTrainer(
        model=patch_nr.normalizing_flow,
        physics=Denoising(UniformNoise(1.0 / 255.0)),
        optimizer=optimizer,
        train_dataloader=patchnr_dataloader,
        device=device,
        losses=[],
        epochs=patchnr_epochs,
        online_measurements=True,
        verbose=verbose,
    )
    trainer.train()
    torch.save(patch_nr.normalizing_flow.state_dict(), "patchnr.pt")
else:
    patch_nr.normalizing_flow.load_state_dict(torch.load(pretrain))

image = dataset[15].unsqueeze(0).float().to(device)


sigma = 0.1
noise_model = GaussianNoise(sigma)
physics = Denoising(device=device, noise_model=noise_model)
data_fidelity = L2()

observation = physics(image)

print(image.shape, observation.shape)

optim_steps = 200
lr_variational_problem = 0.02


def minimize_variational_problem(prior, lam):
    imgs = observation.detach().clone()
    imgs.requires_grad_(True)
    optimizer = torch.optim.Adam([imgs], lr=lr_variational_problem)
    for i in (progress_bar := tqdm(range(optim_steps))):
        optimizer.zero_grad()
        loss = data_fidelity(imgs, observation, physics).mean() + lam * prior.g(imgs)
        loss.backward()
        optimizer.step()
        progress_bar.set_description("Step {} || Loss {}".format(i + 1, loss.item()))
    return imgs.detach()

lam_list = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
psnr_list = [] 
for lam_patchnr in lam_list:
    #lam_patchnr = 0.2

    recon_patchnr = minimize_variational_problem(patch_nr, lam_patchnr)

    psnr_patchnr = PSNR()(recon_patchnr, image)
    psnr_list.append(psnr_patchnr.item())
    print(psnr_patchnr)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)

    ax1.imshow(image[0,0].cpu().numpy(), cmap="gray")
    ax2.imshow(observation[0,0].cpu().numpy(), cmap="gray")
    ax3.imshow(recon_patchnr[0,0].cpu().numpy(), cmap="gray")
    ax3.set_title(f"lam = {lam_patchnr}")
    plt.show()
    """

plt.figure()
plt.plot(lam_list, psnr_list)
plt.show()