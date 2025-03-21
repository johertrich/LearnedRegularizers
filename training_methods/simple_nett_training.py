# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:27:50 2025

@author: JohannesSchwab
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def simple_NETT_training(model, train_dataloader, val_dataloader, device, optimizer, lr, num_epochs):
    criterion = nn.MSELoss()
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, images_gt in tqdm(train_dataloader):
            images, images_gt = images.to(device), images_gt.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images_gt)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, images_gt in val_dataloader:
                images, images_gt = images.to(device), images_gt.to(device)
                outputs = model(images)
                loss = criterion(outputs, images_gt)
                val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")
    
    return model