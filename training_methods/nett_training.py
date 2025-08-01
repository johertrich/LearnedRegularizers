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

def NETT_training(model, train_dataloader, val_dataloader, device, optimizer, lr, num_epochs, save_best = False, weight_dir = None):
    criterion = nn.MSELoss()
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.99, patience=3)
    
    model.to(device)
    min_val_loss = 1.
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, images_gt in tqdm(train_dataloader):
            images, images_gt = images.to(device), images_gt.to(device)
            
            optimizer.zero_grad()
            out_diff = model(images)
            trgt_diff = images-images_gt
            loss = torch.mean(torch.sum((out_diff-trgt_diff)**2, dim = [-1,-2]))+0.01*torch.mean(torch.sum(torch.abs(out_diff-trgt_diff), dim = [-1,-2]))
            loss.backward()
            optimizer.step()
            outputs = images-out_diff

            
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        
        model.eval()
        val_loss = 0.0
        plt.subplot(241)
        plt.imshow(outputs[0,0].detach().cpu(),cmap = 'gray')
        plt.colorbar()
        plt.subplot(242)
        plt.imshow(images[0,0].detach().cpu(),cmap = 'gray')
        plt.colorbar()
        plt.subplot(243)
        plt.imshow(outputs[0,0].detach().cpu()-images_gt[0,0].detach().cpu(),cmap = 'gray')
        plt.colorbar()
        plt.subplot(244)
        plt.imshow(images[0,0].detach().cpu()-images_gt[0,0].detach().cpu(),cmap = 'gray')
        plt.colorbar()
        plt.subplot(245)
        plt.imshow(out_diff[0,0].detach().cpu(),cmap = 'gray')
        plt.colorbar()
        plt.subplot(246)
        plt.imshow(trgt_diff[0,0].detach().cpu(),cmap ='gray')
        plt.colorbar()
        plt.subplot(247)
        plt.imshow(out_diff[0,0].detach().cpu()-trgt_diff[0,0].detach().cpu(),cmap = 'gray')
        plt.colorbar()
        plt.subplot(248)
        plt.imshow(images_gt[0,0].detach().cpu(),cmap ='gray')
        plt.colorbar()
        plt.show()
        
        plt.imshow(out_diff[0,0].detach().cpu())
        plt.colorbar()
        plt.show()
        
        with torch.no_grad():
            for images, images_gt in val_dataloader:
                images, images_gt = images.to(device), images_gt.to(device)
                outputs = model(images)
                loss = criterion(outputs, (images-images_gt))
                val_loss += loss.item()
                # plt.subplot(121)
                # plt.imshow(outputs[0,0].detach().cpu()-images[0,0].detach().cpu())
                # plt.subplot(122)
                # plt.imshow(outputs[0,0].detach().cpu())
                # plt.show()
        
        val_loss /= len(val_dataloader)
        if val_loss < min_val_loss:
            if save_best == True:
                torch.save(model.state_dict(), weight_dir)
                print('new min val loss: saving weights')
            min_val_loss = val_loss
            
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")
    
    return model
