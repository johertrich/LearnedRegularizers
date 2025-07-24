# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:23:47 2025

@author: JohannesSchwab
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class NETT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(NETT, self).__init__()
        
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.bottleneck = conv_block(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.decoder4(torch.cat([self.upconv4(bottleneck), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([self.upconv3(dec4), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self.upconv1(dec2), enc1], dim=1))
        
        return x+self.final(dec1)
    
    def regularizer(self,x):
        return torch.sum(self.forward(x)**2,dim = [1,2,3])
    
    def grad(self,x,create_graph=True):
        r"""
        Calculate the gradient of the potential function.

        :param torch.Tensor x: Input tensor of shape ``(B, C, H, W)``.
        """
        x = x.requires_grad_(True)
        out = self.regularizer(x)
        return torch.autograd.grad(
            outputs=out,
            inputs=x,
            grad_outputs=torch.ones_like(out),
            create_graph=create_graph,
            only_inputs=True,
        )[0]
    
    def g(self, x):
        return self.regularizer(self,x)


class simpleNETT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(simpleNETT, self).__init__()
        self.positivity = False
        
        def conv_block(in_c, out_c,last_activation = True):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode = 'reflect'),
                nn.CELU(alpha = 10, inplace = True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, padding_mode = 'reflect'),
                nn.CELU(alpha = 10, inplace = True)
            )
        
        self.b1 = conv_block(in_channels, 64)
        self.b2 = conv_block(64, 64)
        self.b3 = conv_block(64, 64)
        self.b4 = conv_block(64, 64)
        self.b5 = conv_block(64, 64)
        self.b6 = conv_block(64, 64)
        #self.b7 = conv_block(128, 128)
        #self.b8 = conv_block(128, 128)
        #self.down1 = nn.AvgPool2d(3,stride = 2, padding = 1)
        #self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1, padding = 0, padding_mode = 'reflect')
        
    def forward(self, x):
        enc1 = self.b1(x)
        enc2 = self.b2(enc1)
        #enc_down2 = self.down1(enc2)
        enc3 = self.b3(enc2)
        enc4 = self.b4(enc3)
        #enc_up4 = self.up1(enc4)
        enc5 = self.b5(enc4)
        enc6 = self.b6(enc5)
        #enc7 = self.b7(enc6)
        #enc8 = self.b8(enc7)
        x_out = self.final(enc6)
       # x_out = x-x_out
        
        return x_out
    
    def regularizer(self,x):
        if self.positivity == True:
            return torch.sum(self.forward(x)**2,dim = [1,2,3])+torch.sum(torch.nn.functional.relu(-x[x<0]))
        elif self.positivity == False:
            return torch.sum(self.forward(x)**2,dim = [1,2,3])
            
    
    def grad(self,x,create_graph=True):
        r"""
        Calculate the gradient of the potential function.

        :param torch.Tensor x: Input tensor of shape ``(B, C, H, W)``.
        """
        x = x.requires_grad_(True)
        out = self.regularizer(x)
        return torch.autograd.grad(
            outputs=out,
            inputs=x,
            grad_outputs=None,#torch.ones_like(out),
            create_graph=create_graph,
            only_inputs=True,
        )[0]
    
    def g(self, x):
        return self.regularizer(x)
