# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 10:02:24 2025

@author: Sebastian
"""

import torch
import torch.nn as nn
from deepinv.optim import Prior
from deepinv.models.GSPnP import GSDRUNet

class LSR(Prior):
    def __init__(
        self,
        device="cpu",
        pretrained=None,
    ):
        super(NETT, self).__init__()

        self.model = GSDRUNet(
        alpha=1.0,
        in_channels=1,
        out_channels=1,
        nb=2,
        nc=[64, 128, 256, 512],
        act_mode="E",
        pretrained='download',
        device=device,
        )

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))
    
    def grad(self,x):
        return self.model.potential_grad(x,5)
    
    def g(self, x):
        return self.model.potential(x,5)
