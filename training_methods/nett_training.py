# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:27:50 2025

@author: JohannesSchwab
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from evaluation import reconstruct_nmAPG
from deepinv.loss.metric import PSNR


def NETT_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    device,
    lr,
    num_epochs,
    lr_decay=0.99,
    p=0.5,
    save_best=False,
    weight_dir=None,
    logger=None,
    validation_epochs=10,
    dynamic_range_psnr=False,
):
    optimizer = optim.Adam(regularizer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    if dynamic_range_psnr:
        psnr = PSNR(max_pixel=None)
    else:
        psnr = PSNR()

    regularizer.to(device)
    best_val_psnr = 1.0

    for epoch in range(num_epochs):
        regularizer.train()
        train_loss = 0.0
        for images_gt in tqdm(train_dataloader):
            images_gt = images_gt.to(device)
            images = physics.A_dagger(physics(images_gt))
            inds = torch.rand((images_gt.shape[0],), device=device) > p
            images[inds] = images_gt[inds]

            optimizer.zero_grad()
            out_diff = regularizer(images)
            trgt_diff = images - images_gt
            loss = torch.mean(
                torch.sum((out_diff - trgt_diff) ** 2, dim=[-1, -2])
            ) + 0.01 * torch.mean(
                torch.sum(torch.abs(out_diff - trgt_diff), dim=[-1, -2])
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}")
        if logger is not None:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}")

        if (epoch + 1) % validation_epochs == 0:
            regularizer.eval()
            with torch.no_grad():
                val_loss_epoch = 0
                val_psnr_epoch = 0
                for x_val in tqdm(
                    val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"
                ):
                    x_val = x_val.to(device).to(torch.float32)
                    y_val = physics(x_val)
                    x_val_noisy = physics.A_dagger(y_val)

                    x_recon_val = reconstruct_nmAPG(
                        y_val,
                        physics,
                        data_fidelity,
                        regularizer,
                        lmbd,
                        1e-1,
                        1000,
                        1e-4,
                        verbose=False,
                    )
                    new_psnr = psnr(x_recon_val, x_val).mean().item()
                    if new_psnr <= 0:
                        print(f"Warning: Negativ PSNR occured {new_psnr}")
                    val_psnr_epoch += new_psnr
                mean_val_psnr = val_psnr_epoch / len(val_dataloader)
                print_str = f"[Epoch {epoch+1}] PSNR: {mean_val_psnr:.2f}"
                print(print_str)

                if save_best and mean_val_psnr > best_val_psnr:
                    torch.save(regularizer.state_dict(), weight_dir)

                if logger is not None:
                    logger.info(print_str)

                # ---- Save best regularizer based on validation PSNR ----
                if mean_val_psnr > best_val_psnr:
                    print("Updated best PSNR")
                    best_val_psnr = mean_val_psnr

        scheduler.step()
    if save_best:
        regularizer.load_state_dict(torch.load(weight_dir))

    return regularizer
