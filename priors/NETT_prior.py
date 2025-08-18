# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:23:47 2025

@author: JohannesSchwab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NETT(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, hidden_channels=64, padding_mode="reflect"
    ):
        super(NETT, self).__init__()
        self.positivity = False

        def conv_block(in_c, out_c, last_activation=True):
            return nn.Sequential(
                nn.Conv2d(
                    in_c, out_c, kernel_size=3, padding=1, padding_mode=padding_mode
                ),
                nn.CELU(alpha=10, inplace=True),
                nn.Conv2d(
                    out_c, out_c, kernel_size=3, padding=1, padding_mode=padding_mode
                ),
                nn.CELU(alpha=10, inplace=True),
            )

        self.b1 = conv_block(in_channels, hidden_channels)
        self.b2 = conv_block(hidden_channels, hidden_channels)
        self.b3 = conv_block(hidden_channels, hidden_channels)
        self.b4 = conv_block(hidden_channels, hidden_channels)
        self.b5 = conv_block(hidden_channels, hidden_channels)
        self.b6 = conv_block(hidden_channels, hidden_channels)
        # self.b7 = conv_block(128, 128)
        # self.b8 = conv_block(128, 128)
        # self.down1 = nn.AvgPool2d(3,stride = 2, padding = 1)
        # self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.final = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        enc1 = self.b1(x)
        enc2 = self.b2(enc1)
        # enc_down2 = self.down1(enc2)
        enc3 = self.b3(enc2)
        enc4 = self.b4(enc3)
        # enc_up4 = self.up1(enc4)
        enc5 = self.b5(enc4)
        enc6 = self.b6(enc5)
        # enc7 = self.b7(enc6)
        # enc8 = self.b8(enc7)
        x_out = self.final(enc6)
        # x_out = x-x_out

        return x_out

    def regularizer(self, x):
        if self.positivity == True:
            return torch.sum(self.forward(x) ** 2, dim=[1, 2, 3]) + torch.sum(
                torch.nn.functional.relu(-x[x < 0])
            )
        elif self.positivity == False:
            return torch.sum(self.forward(x) ** 2, dim=[1, 2, 3])

    def grad(self, x, get_energy=False):
        with torch.enable_grad():
            x_ = x.clone()
            x_.requires_grad_(True)
            z = torch.sum(self.g(x_))
            grad = torch.autograd.grad(z, x_, create_graph=True, only_inputs=True)[0]
        if get_energy:
            return z, grad
        return grad

    def g(self, x):
        return self.regularizer(x)
