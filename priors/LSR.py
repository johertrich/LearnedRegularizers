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
        nc=[64, 128, 256, 512],
        pretrained_denoiser=True,
        alpha=1.0,
        sigma=0.03,
    ):
        super(LSR, self).__init__()

        self.model = GSDRUNet(
            alpha=alpha,
            in_channels=1,
            out_channels=1,
            nb=2,
            nc=nc,
            act_mode="s",
            pretrained="download" if pretrained_denoiser else None,
            device=device,
        )

        self.model.detach=False

        self.sigma=sigma

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))

    def grad(self, x):
        return self.model.potential_grad(x, self.sigma)

    def g(self, x):
        return self.model.potential(x, self.sigma)
