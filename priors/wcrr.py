"""
Implements the (Weakly) Convex Ridge Regularizer, which is a special case of a Field-of-Experts regularizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from deepinv.optim import Prior


class ZeroMean(nn.Module):
    """Enforcing zero mean on the filters improves performances"""

    def forward(self, x):
        return x - torch.mean(x, dim=(1, 2, 3), keepdim=True)


class WCRR(Prior):
    def __init__(
        self,
        sigma,  # scaling constant (we set it always to 0.1)
        weak_convexity,  # set to 0.0 for CRR and to 1.0 for WCRR
        tanh=False,  # whether to use tanh instead of smoothed l1 as potential
        nb_channels=[1, 4, 8, 64],  # number channels in the multiconv
        filter_sizes=[5, 5, 5],  # filter sizes of the convolution
        device="cuda" if torch.cuda.is_available() else "cpu",  # device
        pretrained=None,  # if a str is given here, the weights from the corresponding path will be loaded
    ):
        super(WCRR, self).__init__()

        self.nb_filters = nb_channels[-1]
        self.filter_size = sum(filter_sizes) - len(filter_sizes) + 1
        self.filters = nn.Sequential(
            *[
                nn.Conv2d(
                    nb_channels[i],
                    nb_channels[i + 1],
                    filter_sizes[i],
                    padding=filter_sizes[i] // 2,
                    bias=False,
                )
                for i in range(len(filter_sizes))
            ]
        )
        P.register_parametrization(self.filters[0], "weight", ZeroMean())

        self.dirac = torch.zeros(
            1, 1, 2 * self.filter_size - 1, 2 * self.filter_size - 1
        )
        self.dirac[0, 0, self.filter_size - 1, self.filter_size - 1] = 1.0

        self.scaling = nn.Parameter(
            torch.log(torch.tensor(2.0) / sigma) * torch.ones(1, self.nb_filters, 1, 1)
        )
        self.beta = nn.Parameter(torch.tensor(4.0))
        self.weak_cvx = weak_convexity
        self.tanh = tanh

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))

    def smooth_l1(self, x):
        if self.tanh:
            x_abs = torch.abs(x)
            return torch.log((torch.exp(x - x_abs) + torch.exp(-x - x_abs)) / 2) + x_abs
        return torch.clip(x**2, 0.0, 1.0) / 2 + torch.clip(torch.abs(x), 1.0) - 1.0

    def grad_smooth_l1(self, x):
        if self.tanh:
            return torch.tanh(x)
        else:
            return torch.clip(x, -1.0, 1.0)

    def get_conv_lip(self):
        impulse = self.filters(self.dirac)
        for filt in reversed(self.filters):
            impulse = F.conv_transpose2d(impulse, filt.weight, padding=filt.padding)
        return torch.fft.fft2(impulse, s=[256, 256]).abs().max()

    def conv(self, x):
        x = x / torch.sqrt(self.get_conv_lip())
        return self.filters(x)

    def conv_transpose(self, x):
        x = x / torch.sqrt(self.get_conv_lip())
        for filt in reversed(self.filters):
            x = F.conv_transpose2d(x, filt.weight, padding=filt.padding)
        return x

    def grad(self, x, get_energy=False):
        grad = self.conv(x)
        grad = grad * torch.exp(self.scaling)
        if get_energy:
            reg = (
                self.smooth_l1(torch.exp(self.beta) * grad) * torch.exp(-self.beta)
                - self.smooth_l1(grad) * self.weak_cvx
            )
            reg = reg * torch.exp(-2 * self.scaling)
            reg = reg.sum(dim=(1, 2, 3))
        grad = (
            self.grad_smooth_l1(torch.exp(self.beta) * grad)
            - self.grad_smooth_l1(grad) * self.weak_cvx
        )
        grad = grad * torch.exp(-self.scaling)
        grad = self.conv_transpose(grad)
        if get_energy:
            return reg, grad
        return grad

    def g(self, x):
        reg = self.conv(x)
        reg = reg * torch.exp(self.scaling)
        reg = (
            self.smooth_l1(torch.exp(self.beta) * reg) * torch.exp(-self.beta)
            - self.smooth_l1(reg) * self.weak_cvx
        )
        reg = reg * torch.exp(-2 * self.scaling)
        reg = reg.sum(dim=(1, 2, 3))
        return reg

    def _apply(self, fn):
        self.dirac = fn(self.dirac)
        return super()._apply(fn)
