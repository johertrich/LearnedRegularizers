# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:34:22 2025

@author: JohannesSchwab
"""
import torch
import torch.nn.functional as F
import numpy as np

class NETT_transform(object):
    """Rescale the image in a sample to a given size.

    Args:
        percentage (float): Number between 0 and 1 that defines the ratio between distorted and ground truth images.
        physics: defines the physics of the problem
    """

    def __init__(self, percentage,physics):
        self.percentage = percentage
        self.physics = physics

    def __call__(self, sample):
        image = sample.float()
        image = circular_mask(image)
        if torch.rand(1) > self.percentage:
            img = image
        else:
            img = self.physics.A_dagger(self.physics.A(image.unsqueeze(1))).squeeze(1)


        return img, image
    
def circular_mask(image):
    """
    Applies a circular mask to an image using PyTorch, setting everything outside the circle to zero.
    
    Parameters:
        image (torch.Tensor): The input image as a PyTorch tensor of shape (C, H, W).
    
    Returns:
        torch.Tensor: The masked image with everything outside the circle set to zero.
    """
    _, height, width = image.shape
    center = (width // 2, height // 2)
    radius = min(center)
    
    # Create coordinate grid
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    
    # Create circular mask
    mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius ** 2
    
    # Apply mask
    masked_image = image * mask.unsqueeze(0)  # Broadcast mask to all channels
    
    return masked_image