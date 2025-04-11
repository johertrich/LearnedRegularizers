# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26th 2025

@author: Zakobian
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def WGAN_loss(regularizer, images, images_gt,mu=10):
    """Calculates the gradient penalty loss for WGAN GP"""
    real_samples=images_gt
    fake_samples=images
    
    with torch.enable_grad():
        alpha = torch.rand(real_samples.size(0), 1, 1, 1).type_as(real_samples)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        net_interpolates = regularizer.g(interpolates)
        fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).type_as(real_samples).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=net_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

    gradients = gradients.view(gradients.size(0), -1)
    # print(model(real_samples).mean()-model(fake_samples).mean(),self.mu*(((gradients.norm(2, dim=1) - 1)) ** 2).mean())
    loss = regularizer.g(real_samples).mean() - regularizer.g(fake_samples).mean()+mu*(((gradients.norm(2, dim=1) - 1)) ** 2).mean()
    return loss

def estimate_lmbd(dataset,physics,device):
    lmbd=None
    if dataset is None: lmbd=1.0
    else: 
        residual = 0.0
        for x in tqdm(dataset):
            if isinstance(x, list):
                x = x[0]

            if device == "mps":
                x = x.to(torch.float32).to(device)
            else:
                x = x.to(device).to(torch.float)
            y = physics(x) ##Ax+e
            noise = y - physics.A(x)
            residual += torch.norm(physics.A_adjoint(noise),dim=(-2,-1)).mean()
            # residual += torch.sqrt(torch.sum((x_noisy-x)**2))
        lmbd = residual/(len(dataset))
    print('Estimated lambda: ' + str(lmbd))
    return lmbd

def simple_ar_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=25,
    lr=1e-3,
    lr_decay=0.95,
    device="cuda" if torch.cuda.is_available() else "cpu",
    mu = 10.0,
):
    adversarial_loss = WGAN_loss
    regularizer.to(device)
    # optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    optimizer = torch.optim.RMSprop(regularizer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay, patience=2)
    
    
    for epoch in range(epochs):
        loss_vals = []
        for x in tqdm(train_dataloader):
            optimizer.zero_grad()
            if isinstance(x, list):
                x = x[0]
            if device == "mps":
                x = x.to(torch.float32).to(device)
            else:
                x = x.to(device).to(torch.float)
            y = physics(x)
            x_noisy = physics.A_dagger(y)
            loss = adversarial_loss(regularizer, x_noisy, x, mu)
        
            mean_loss = torch.mean(loss)
            mean_loss.backward()
            optimizer.step()
            loss_vals.append(mean_loss.item())
        print(
            "Average training loss in epoch {0}: {1:.2E}".format(
                epoch + 1, np.mean(loss_vals)
            )
        )

        with torch.no_grad():
            loss_vals = []
            for x in tqdm(val_dataloader):
                if isinstance(x, list):
                    x = x[0]
                if device == "mps":
                    x = x.to(torch.float32).to(device)
                else:
                    x = x.to(device).to(torch.float)
                y = physics(x)
                x_noisy = physics.A_dagger(y)
                loss = adversarial_loss(regularizer, x_noisy, x, mu)
                mean_loss = torch.mean(loss)
                
                loss_vals.append(mean_loss.item())
            
        scheduler.step(np.mean(loss_vals))
            
        print(
            "Average validation loss in epoch {0}: {1:.2E}".format(
                epoch + 1, np.mean(loss_vals)
            )
        )

        print("Learning rate: ", scheduler.get_last_lr()[0])
