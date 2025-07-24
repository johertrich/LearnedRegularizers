"""
Evaluation Script for the FBPUnet

@author: Alex
"""

import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from tqdm import tqdm 
from operators import get_evaluation_setting
import numpy as np 
import matplotlib.pyplot as plt 

import yaml 
from deepinv.loss.metric import PSNR

from supervised_training.learned_primal_dual import PrimalDualNet

problem = "CT"
# "LocalAR_{trainingmethod}_p={patchszie}_{training_data}.pt"

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

with open('supervised_training/lpd_config.yaml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

print(cfg)

model = PrimalDualNet(image_width=362, num_angles=60, n_iter=cfg["model_params"]["n_iter"], op=physics.A, op_adj=physics.A_dagger, op_init=physics.A_dagger, 
                 n_primal=cfg["model_params"]["n_primal"], n_dual=cfg["model_params"]["n_dual"],
                 use_sigmoid=cfg["model_params"]["use_sigmoid"], n_layer=cfg["model_params"]["n_layer"], 
                 internal_ch=cfg["model_params"]["internal_ch"], kernel_size=cfg["model_params"]["kernel_size"],
                 batch_norm=cfg["model_params"]["batch_norm"], lrelu_coeff=cfg["model_params"]["lrelu_coeff"])
model.load_state_dict(torch.load("supervised_training/lpdnet.pt"))
model.eval()
model.to(device)

print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
psnr = PSNR(max_pixel=None)

with torch.no_grad():
    ## Evaluate on the test set
    psnrs = []
    for i, x in tqdm(enumerate(dataloader), total=len(dataloader)):

        x = x.to(device)
        y = physics(x)
        x_fbp = physics.A_dagger(y)

        x_pred = model(y)
        
        #fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        #ax1.imshow(x[0,0].cpu().numpy(), cmap="gray")
        #ax2.imshow(x_fbp[0,0].cpu().numpy(), cmap="gray")
        #ax3.imshow(x_pred[0,0].cpu().numpy(), cmap="gray")
        #plt.show()

        psnrs.append(psnr(x_pred, x).squeeze().item())


print("PSNR: ", np.mean(psnrs))