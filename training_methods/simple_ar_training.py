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
from evaluation import evaluate
from dataset import get_dataset
from torchvision.transforms import CenterCrop, RandomCrop
from deepinv.utils import patch_extractor
from torch.utils.data import DataLoader


def WGAN_loss(regularizer, images, images_gt,mu=10):
    """Calculates the gradient penalty loss for WGAN GP"""
    real_samples=images_gt
    fake_samples=images
    
    # with torch.enable_grad():
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).type_as(real_samples)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    net_interpolates = regularizer.g(interpolates)
    if (len(net_interpolates.shape) == 1):
        net_interpolates = net_interpolates.view(-1, 1)
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
    
    if physics.__class__.__name__ == "Tomography":
        val_data = val_dataloader.dataset
    else:
        val_data = get_dataset("BSD68", transform=CenterCrop(256))
    regularizer.train()
    NAG_step_size = 1e-1  # step size in NAG
    NAG_max_iter = 500  # maximum number of iterations in NAG
    NAG_tol = 1e-4  # tolerance for therelative error (stopping criterion)
    only_first = False
    def eval_routine():
        mean_psnr, x_out, y_out, recon_out = evaluate(
                    physics=physics,
                    data_fidelity=data_fidelity,
                    dataset=val_data,  # Access only the dataset of the dataloader
                    regularizer=regularizer,
                    lmbd=lmbd,
                    NAG_step_size=NAG_step_size,
                    NAG_max_iter=NAG_max_iter,
                    NAG_tol=NAG_tol,
                    only_first=only_first,
                    device=device,
                    verbose=False,
                )
        for p in regularizer.parameters():
            p.requires_grad_(True)
        return mean_psnr, x_out, y_out, recon_out
    mean_psnr, x_out, y_out, recon_out = eval_routine()
    print("Initial reconstruction: ", mean_psnr)
    for epoch in range(epochs):
        loss_vals = []
        regularizer.train()
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

        # with torch.no_grad():
        
        
        loss_vals = []
        # for x in tqdm(val_dataloader):
        
        mean_psnr, x_out, y_out, recon_out = eval_routine()
        # print(mean_psnr)
        loss_vals.append(mean_psnr.item())
            
        scheduler.step(np.mean(loss_vals))
            
        # print(
        #     "Average validation loss in epoch {0}: {1:.2E}".format(
        #         epoch + 1, np.mean(loss_vals)
        #     )
        # )

        print("Learning rate: ", scheduler.get_last_lr()[0])




# Training function for the LocalAR
def simple_lar_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_data,
    val_data,
    patch_size,
    epochs=25,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    mu = 10.0,
    batch_size=128
):
    adversarial_loss = WGAN_loss
    regularizer.to(device)
    optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100. )
    
    regularizer.train()
    NAG_step_size = 1e-2  # step size in NAG
    NAG_max_iter = 100  # maximum number of iterations in NAG
    NAG_tol = 1e-4  # tolerance for therelative error (stopping criterion)
    only_first = True
    def eval_routine():
        mean_psnr, mean_psnr_dr, x_out, y_out, recon_out = evaluate(
                    physics=physics,
                    data_fidelity=data_fidelity,
                    dataset=val_data,  # Access only the dataset of the dataloader
                    regularizer=regularizer,
                    lmbd=lmbd,
                    NAG_step_size=NAG_step_size,
                    NAG_max_iter=NAG_max_iter,
                    NAG_tol=NAG_tol,
                    only_first=only_first,
                    device=device,
                    verbose=False,
                )
        for p in regularizer.parameters():
            p.requires_grad_(True)
        return mean_psnr, x_out, y_out, recon_out
    mean_psnr, x_out, y_out, recon_out = eval_routine()

    print("PSNR of initial model: ", mean_psnr)
    for epoch in tqdm(range(epochs)):
        loss_vals = []
        regularizer.train()

        train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
        for x in train_dataloader:
            optimizer.zero_grad()
            if isinstance(x, list):
                x = x[0]
            if device == "mps":
                x = x.to(torch.float32).to(device)
            else:
                x = x.to(device).to(torch.float)
            y = physics(x)
            x_noisy = physics.A_dagger(y)

            ### patch-based training
            x_patches, linear_inds = patch_extractor(x, n_patches=batch_size, patch_size=patch_size)
            x_patches = x_patches.squeeze(0)

            _, C, _, _ = x_noisy.shape
            imgs = x_noisy.reshape(1, -1)
            x_noisy_patches = imgs.view(1, -1)[:, linear_inds]
            x_noisy_patches = x_noisy_patches.reshape(batch_size, C, patch_size, patch_size)

            loss = adversarial_loss(regularizer.cnn, x_noisy_patches, x_patches, mu)

            mean_loss = torch.mean(loss)
            mean_loss.backward()
            optimizer.step()
            loss_vals.append(mean_loss.item())
        print(
            "Average training loss in epoch {0}: {1:.2E}".format(
                epoch + 1, np.mean(loss_vals)
            )
        )
        
        loss_vals = []
        
        
        scheduler.step()    
        
        print("Learning rate: ", scheduler.get_last_lr()[0])
