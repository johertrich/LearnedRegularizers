import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

from deepinv.optim import EPLL
from deepinv.physics import GaussianNoise, Denoising, Inpainting
from deepinv.loss.metric import PSNR
from deepinv.utils import plot
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.datasets import PatchDataset
from deepinv.optim.data_fidelity import L2

from dataset import get_dataset
from torchvision.transforms import RandomCrop


device = "cuda" if torch.cuda.is_available() else "cpu"



dataset = get_dataset("BSDS500_gray", test=False, transform=RandomCrop(256))

train_imgs = [] 
for i in range(20):
    train_imgs.append(dataset[i].unsqueeze(0).float())

train_imgs = torch.concat(train_imgs)

print(train_imgs.shape)
patch_size = 8
patchnr_batch_size = 256
model_epll = EPLL(channels=train_imgs.shape[1], patch_size=patch_size, device=device, pretrained=None)

train_dataset = PatchDataset(train_imgs, patch_size=patch_size, transforms=None)
patchnr_dataloader = DataLoader(
    train_dataset,
    batch_size=patchnr_batch_size,
    shuffle=True,
    drop_last=True,
)

model_epll.GMM.fit(patchnr_dataloader, verbose=True, max_iters=10)
torch.save(model_epll.GMM.state_dict(),"gmm.pt")
