"""
Created on Wed Feb 26 2025

@author: Zakobian
"""
from deepinv.optim import Prior
import torch
import torch.nn as nn

class linear_ICNN(nn.Module):
    r"""
    Convolutional Input Convex Neural Network (ICNN).
    Unlike the deepinv ICNN, this model is linear without the quadratic layers (hoping for stabler training).
    """

    def __init__(
        self,
        in_channels=3,
        num_filters=64,
        kernel_dim=5,
        num_layers=10,
        strong_convexity=0.5,
        pos_weights=True,
        device="cpu",
    ):
        super(linear_ICNN, self).__init__()
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
                    padding_mode="circular",
                    bias=False,
                    device=device,
                )
                for i in range(self.n_layers)
            ]
        )

        # these layers can have arbitrary weights
        # self.wx_quad = nn.ModuleList(
        #     [
        #         nn.Conv2d(
        #             self.n_in_channels,
        #             self.n_filters,
        #             self.kernel_size,
        #             stride=1,
        #             padding=self.padding,
        #             padding_mode="circular",
        #             bias=False,
        #             device=device,
        #         )
        #         for i in range(self.n_layers + 1)
        #     ]
        # )
        self.wx_lin = nn.ModuleList(
            [
                nn.Conv2d(
                    self.n_in_channels,
                    self.n_filters,
                    self.kernel_size,
                    stride=1,
                    padding=self.padding,
                    padding_mode="circular",
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
            padding_mode="circular",
            bias=False,
            device=device,
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # slope of leaky-relu
        self.negative_slope = 0.2
        self.strong_convexity = strong_convexity

        self.pos_weights = pos_weights
        self.device = device


    def forward(self, x):
        r"""
        Calculate potential function of the ICNN.

        :param torch.Tensor x: Input tensor of shape ``(B, C, H, W)``.
        """
        if self.pos_weights:
            self.zero_clip_weights()
        z = torch.nn.functional.leaky_relu(
            # self.wx_quad[0](x) ** 2 +
            self.wx_lin[0](x),
            negative_slope=self.negative_slope,
        )
        for layer in range(self.n_layers):
            z = torch.nn.functional.leaky_relu(
                self.wz[layer](z)
                # + self.wx_quad[layer + 1](x) ** 2
                + self.wx_lin[layer + 1](x),
                negative_slope=self.negative_slope,
            )
        z = self.final_conv2d(z)
        z_avg = self.pool(z).reshape(-1,1)
        # z_avg = torch.nn.functional.avg_pool2d(z, z.size()[2:]).view(z.size()[0], -1)

        return z_avg + 0.5 * self.strong_convexity * (x**2).sum(
            dim=[1, 2, 3]
        ).reshape(-1, 1)

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


class linearICNNPrior(Prior):
    def __init__(
        self,
        in_channels=3,
        num_filters=64,
        kernel_dim=5,
        num_layers=10,
        strong_convexity=0.5,
        pos_weights=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained=None,
    ):
        super().__init__()
        self.icnn = linear_ICNN(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_dim=kernel_dim,
            num_layers=num_layers,
            strong_convexity=strong_convexity,
            pos_weights=pos_weights,
            device=device,
        )
        self.add_module("ICNN", self.icnn)
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))

    def g(self, x):
        return self.icnn(x)

    def grad(self, x):
        with torch.enable_grad():
            x_ = x.clone()
            x_.requires_grad_(True)
            val = torch.sum(self.g(x_))
            grad = torch.autograd.grad(val, x_, create_graph=True)[0]
        return grad
