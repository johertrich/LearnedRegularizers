"""
Created on Wed Feb 26 2025

@author: Zakobian
"""
from deepinv.optim import Prior
import torch
import torch.nn as nn

class network(nn.Module):
    def __init__(self,in_channels,size=64,kernel_dim=5):
        super(network, self).__init__()

        self.leaky_relu = nn.LeakyReLU()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(kernel_dim, kernel_dim),padding=kernel_dim//2),
            self.leaky_relu,
            nn.Conv2d(16, 32, kernel_size=(kernel_dim, kernel_dim),padding=kernel_dim//2),
            self.leaky_relu,
            nn.Conv2d(32, 32, kernel_size=(kernel_dim, kernel_dim),padding=kernel_dim//2,stride=kernel_dim//2),
            self.leaky_relu,
            nn.Conv2d(32, 64, kernel_size=(kernel_dim, kernel_dim),padding=kernel_dim//2,stride=kernel_dim//2),
            self.leaky_relu,
            nn.Conv2d(64, 64, kernel_size=(kernel_dim, kernel_dim),padding=kernel_dim//2,stride=kernel_dim//2),
            self.leaky_relu,
            nn.Conv2d(64, 128, kernel_size=(kernel_dim, kernel_dim),padding=kernel_dim//2,stride=kernel_dim//2),
            self.leaky_relu
        )
        # size=1024
        self.fc = nn.Sequential(
            nn.Linear(128*(size//(kernel_dim//2)**4)**2, 256),
            self.leaky_relu,
            nn.Linear(256, 1)
        )

    def forward(self, image):
        output = self.convnet(image)
        output = output.view(image.size(0), -1)
        output = self.fc(output)
        return output


class CNNPrior(Prior):
    def __init__(
        self,
        in_channels=3,
        size=64,
        kernel_dim=5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained=None,
    ):
        super().__init__()
        self.nn = network(
            in_channels=in_channels,
            size = 64,
            kernel_dim=kernel_dim,
        )
        self.nn.to(device)
        self.add_module("NN", self.nn)
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))

    def g(self, x):
        return self.nn(x)

    def grad(self, x):
        with torch.enable_grad():
            x_ = x.clone()
            x_.requires_grad_(True)
            val = torch.sum(self.g(x_))
            grad = torch.autograd.grad(val, x_, create_graph=True)[0]
        return grad
