"""
Created on Mon March 10 2025

@author: Yasi Zhang

Convolutional Input Difference-of-Convex Neural Network (IDCNN) proposed in https://arxiv.org/pdf/2502.00240.
In short, IDCNN is the substraction of two ICNNs.
The ICNNs used in IDCNN are linear without the quadratic layers following the original ICNN paper (https://arxiv.org/pdf/1609.07152).

"""
from deepinv.optim import Prior
import torch
import torch.nn as nn

class linear_ICNN(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_filters=64,
        kernel_dim=5,
        num_layers=10,
        pos_weights=True,
        device="cpu",
        padding_mode="zeros",
        beta=100,
    ):
        super().__init__()
        self.n_in_channels = in_channels
        self.n_layers = num_layers
        self.n_filters = num_filters
        self.kernel_size = kernel_dim
        self.padding = (self.kernel_size - 1) // 2
     
        # these layers should have non-negative weights
        self.wz = nn.ModuleList(
            [
                nn.Conv2d(
                    self.n_filters,
                    self.n_filters,
                    self.kernel_size,
                    stride=1,
                    padding=self.padding,
                    padding_mode=padding_mode,
                    bias=False,
                    device=device,
                )
                for i in range(self.n_layers)
            ]
        )
        # these layers can have arbitrary weights
        self.wx_lin = nn.ModuleList(
            [
                nn.Conv2d(
                    self.n_in_channels,
                    self.n_filters,
                    self.kernel_size,
                    stride=1,
                    padding=self.padding,
                    padding_mode=padding_mode,
                    bias=True,
                    device=device,
                )
                for i in range(self.n_layers + 1)
            ]
        )
        # one final conv layer with nonnegative weights
        self.final_conv2d = nn.Conv2d(
            self.n_filters,
            self.n_in_channels,
            self.kernel_size,
            stride=1,
            padding=self.padding,
            padding_mode=padding_mode,
            bias=False,
            device=device,
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.pos_weights = pos_weights
        self.device = device
        self.beta = beta

        self.initialize_weights()


    def forward(self, x):
        r"""
        Calculate potential function of the IDCNN.

        :param torch.Tensor x: Input tensor of shape ``(B, C, H, W)``.
        """
        if self.pos_weights:
            self.zero_clip_weights()
         
        z = torch.nn.functional.softplus(
            self.wx_lin[0](x), beta=self.beta
        )
       
        for layer in range(self.n_layers):
            z = torch.nn.functional.softplus(
                self.wz[layer](z)
                + self.wx_lin[layer + 1](x), beta=self.beta
            )
         
        z = self.final_conv2d(z)
        z_avg = self.pool(z).reshape(-1,1)

        return z_avg  

    def grad(self, x):
        r"""
        Calculate the gradient of the potential function.

        :param torch.Tensor x: Input tensor of shape ``(B, C, H, W)``.
        """
        x = x.requires_grad_(True)
        out = self.forward(x)
        return torch.autograd.grad(
            outputs=out,
            inputs=x,
            grad_outputs=torch.ones_like(out),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

    # a weight initialization routine for the ICNN, with positive weights
    def initialize_weights(self, min_val=0.0, max_val=0.001):
        for layer in range(self.n_layers):
            self.wz[layer].weight.data = min_val + (max_val - min_val) * torch.rand(
                self.n_filters, self.n_filters, self.kernel_size, self.kernel_size
            ).to(self.device)
        self.final_conv2d.weight.data = min_val + (max_val - min_val) * torch.rand(
            1, self.n_filters, self.kernel_size, self.kernel_size
        ).to(self.device)
        return self

    # a zero clipping functionality for the ICNN (set negative weights to 0)
    def zero_clip_weights(self):
        for layer in range(self.n_layers):
            self.wz[layer].weight.data.clamp_(0)
        self.final_conv2d.weight.data.clamp_(0)
        return self


class linearIDCNNPrior_Softplus(Prior):
    def __init__(
        self,
        in_channels=3,
        num_filters=64,
        kernel_dim=5,
        num_layers=10,
        pos_weights=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained=None,
        beta=100,
    ):
        super().__init__()
        self.icnn1 = linear_ICNN(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_dim=kernel_dim,
            num_layers=num_layers,
            pos_weights=pos_weights,
            device=device,
            beta=beta,
        )
        self.icnn2 = linear_ICNN(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_dim=kernel_dim,
            num_layers=num_layers,
            pos_weights=pos_weights,
            device=device,
            beta=beta,
        )
        self.add_module("ICNN1", self.icnn1)
        self.add_module("ICNN2", self.icnn2)
        
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))
        

    def g(self, x):
        return self.icnn1(x) - self.icnn2(x)

    def grad(self, x):
        with torch.enable_grad():
            x_ = x.clone()
            x_.requires_grad_(True)
            val = torch.sum(self.g(x_))
            grad = torch.autograd.grad(val, x_, create_graph=True)[0]
        return grad


if __name__ == "__main__":
    # test the IDCNN
    import numpy as np
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    idcnn = linear_ICNN(device=device)
    x = torch.tensor(np.random.randn(1, 3, 128, 128), dtype=torch.float32, device=device)
    x.requires_grad = True
    y = idcnn(x)
    grad = idcnn.grad(x)
    print(y.shape)
    print(grad.shape)

    prior = linearIDCNNPrior(device=device)
    x.requires_grad = True
    y = prior.g(x)
    grad = prior.grad(x)
    print(y.shape)
    print(grad.shape)
    print("Done")
