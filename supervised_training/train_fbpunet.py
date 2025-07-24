# -*- coding: utf-8 -*-
"""
Training of post-processing UNet 

@author: Alexander
"""

import sys 
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from dataset import get_dataset
from tqdm import tqdm 
import numpy as np 
import yaml 

from operators import get_operator
from supervised_training.unet import get_unet_model

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "CT"
dataset = get_dataset("LoDoPaB", test=False, transform=None)
physics , data_fidelity = get_operator(problem, device)
lmbd = 1.0
test_ratio = 0.01

# splitting in training and validation set
test_len = int(len(dataset) * test_ratio)
train_len = len(dataset) - test_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])

batch_size = 16
shuffle = True

cfg = {
    "lr": 1e-4,
    "num_epochs": 1000,
    "model_params": {
        "use_norm": False,
        "scales": 5,
        "use_sigmoid": False,
        "skip": 16,
        "channels": (16, 32, 64, 64, 128, 128)
    },
}


model = get_unet_model(use_norm=cfg["model_params"]["use_norm"], 
                        scales=cfg["model_params"]["scales"],
                        use_sigmoid=cfg["model_params"]["use_sigmoid"], 
                        skip=cfg["model_params"]["skip"],
                        channels=cfg["model_params"]["channels"])
model.train()
model.to(device)

train_dl = torch.utils.data.DataLoader(train_set, batch_size=16)
val_dl = torch.utils.data.DataLoader(val_set, batch_size=1)

print("Length of training set: ", len(train_set))
print("Length of val set: ", len(val_set))

optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg["num_epochs"], eta_min=1e-3/100)

min_loss = 100

for epoch in range(cfg["num_epochs"]):

    model.train()
    for idx, batch in (progress_bar := tqdm(enumerate(train_dl), total=len(train_dl))):
        optimiser.zero_grad()
        x = batch.to(device)
        y = physics(x)
        x_fbp = physics.A_dagger(y)

        x_pred = model(x_fbp)


        loss = torch.mean((x_pred - x)**2)
        loss.backward()

        optimiser.step()


        progress_bar.set_description(
                "Epoch {} || Step {} || Loss {:.7f} ".format(epoch+1,
                    idx + 1, loss.item()
                )
            )

    model.eval() 
    scheduler.step() 
    with torch.no_grad():
        mean_loss = [] 
        for batch in val_dl:
            x = batch.to(device)
            y = physics(x)
            x_fbp = physics.A_dagger(y)

            x_pred = model(x_fbp)


            loss = torch.mean((x_pred - x)**2)

            mean_loss.append(loss.item())

    print("Validation Loss: ", np.mean(mean_loss))

    if np.mean(mean_loss) < min_loss:
        min_loss = np.mean(mean_loss)
        print("new min loss: save model")
        with open("supervised_training/fbpunet_config.yaml", "w") as f:
            yaml.dump(cfg, f)
        torch.save(model.state_dict(), "supervised_training/fbpunet.pt")
