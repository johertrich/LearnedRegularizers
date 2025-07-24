"""
Evaluation Script for the FBPUnet

@author: Alex
"""

import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from operators import get_evaluation_setting

import torch
from tqdm import tqdm 
import numpy as np 
import matplotlib.pyplot as plt 

import yaml 
from deepinv.loss.metric import PSNR

from supervised_training.unet import get_unet_model

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

with open('supervised_training/fbpunet_config.yaml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

print(cfg)

model = get_unet_model(use_norm=cfg["model_params"]["use_norm"], 
                        scales=cfg["model_params"]["scales"],
                        use_sigmoid=cfg["model_params"]["use_sigmoid"], 
                        skip=cfg["model_params"]["skip"],
                        channels=cfg["model_params"]["channels"])
model.load_state_dict(torch.load("supervised_training/fbpunet.pt"))
model.eval()
model.to(device)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
psnr = PSNR(max_pixel=None)

with torch.no_grad():
    ## Evaluate on the test set
    psnrs = []
    for i, x in tqdm(enumerate(dataloader), total=len(dataloader)):

        x = x.to(device)
        y = physics(x)
        x_fbp = physics.A_dagger(y)

        x_pred = model(x_fbp)
        
        #fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        #ax1.imshow(x[0,0].cpu().numpy(), cmap="gray")
        #ax2.imshow(x_fbp[0,0].cpu().numpy(), cmap="gray")
        #ax3.imshow(x_pred[0,0].cpu().numpy(), cmap="gray")
        #plt.show()

        psnrs.append(psnr(x_pred, x).squeeze().item())


print("PSNR: ", np.mean(psnrs))