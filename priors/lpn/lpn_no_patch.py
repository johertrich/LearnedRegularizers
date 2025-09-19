"""
Created on Wed Mar 5 2025

@author: ZhenghanFang
"""

import torch
from deepinv.models.utils import test_pad
from deepinv.optim import Prior

from .invert_model import invert
from .prior import evaluate_prior


class LPNPrior(Prior):
    def __init__(
        self,
        in_dim=1,
        hidden=256,
        beta=100.0,
        alpha=1e-6,
        pretrained=None,
        clip=True,
    ):
        """
        Args:
            in_dim: int, input dimension
            hidden: int, hidden channel size
            beta: float, beta in softplus
            alpha: float, strongly-convex parameter
            model_name: str, select lpn model
        """
        super().__init__()
        self.lpn = LPN(in_dim, hidden, beta, alpha)
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location="cpu"))

        self.clip = clip

    def g(self, x: torch.Tensor, inv_alg="cvx_cg", **kwargs) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, shape (B, *), normalized to [0, 1]

        Returns:
            torch.Tensor, shape (B, ), prior at each sample
        """
        x = x * 2 - 1  # convert to [-1, 1]
        x_np = x.detach().cpu().numpy()
        prior = evaluate_prior(x_np, self.lpn, inv_alg=inv_alg, **kwargs)["p"]
        return torch.tensor(prior, device=x.device)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the gradient of the regularizer at x by f^{-1}(x) - x, where
        f^{-1} is the inverse of LPN.

        Args:
            x: torch.Tensor, shape (B, *), normalized to [0, 1]

        Returns:
            torch.Tensor, shape (B, *)
        """
        x = x * 2 - 1  # convert to [-1, 1]
        x_np = x.detach().cpu().numpy()
        y = invert(x_np, self.lpn, inv_alg="cvx_cg")
        y = torch.tensor(y, device=x.device)
        return y - x

    def prox(self, x, *args, **kwargs):
        x = x * 2 - 1  # convert to [-1, 1]
        if x.shape[-2] % 8 == 0 and x.shape[-1] % 8 == 0:
            out = self.lpn(x)
        else:
            out = test_pad(self.lpn, x, 8)
        # convert back to [0, 1]
        out = (out + 1) / 2
        # clipping
        if self.clip:
            out = torch.clamp(out, 0, 1)
        return out

    def forward(self, x):
        return self.prox(x)

    def wclip(self):
        self.lpn.wclip()

    @property
    def img_size(self):
        return self.lpn.img_size


"""Learned proximal networks for input size of 64x64.
The base ICNN (`_scalar` method) takes input in the range of [-1, 1].
The scalar and forward methods take input in the range of [0, 1], and convert it to [-1, 1] before passing to the base ICNN.
The return of forward is in the range of [0, 1].

Modified from https://github.com/ZhenghanFang/learned-proximal-networks/blob/main/lpn/networks/lpn_128.py
"""

import numpy as np
import torch
from torch import nn


class LPN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden,
        beta,
        alpha,
    ):
        super().__init__()

        self.hidden = hidden
        self.lin = nn.ModuleList(
            [
                nn.Conv2d(in_dim, hidden, 3, bias=True, stride=1, padding=1),  # 64
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=1, padding=1),  # 32
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=1, padding=1),  # 32
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=1, padding=1),  # 16
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=1, padding=1),  # 16
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=1, padding=1),  # 8
                nn.Conv2d(hidden, 64, 7, bias=False, stride=1, padding=3),  # 8
                nn.Conv2d(64, 1, 1),
            ]
        )

        self.res = nn.ModuleList(
            [
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1),  # 32
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1),  # 32
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1),  # 16
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1),  # 16
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1),  # 8
                nn.Conv2d(in_dim, 64, 7, stride=1, padding=3),  # 8
            ]
        )

        self.pool = nn.ModuleList(
            [
                nn.AvgPool2d(2),
                nn.AvgPool2d(2),
                nn.AvgPool2d(2),
            ]
        )
        self.pool_inds = [0, 2, 4]

        self.act = nn.Softplus(beta=beta)
        self.alpha = alpha

        self.img_size = 64  # expected image size

        # init weights
        self.init_weights(-10, 0.1)

    def scalar(self, x):
        """Scalar function. Convex. ICNN.
        Args:
            x: (b, c, w, h), normalized to [-1, 1]
        Note:
            The default input size is 64x64. For other sizes, the network extrapolates.
        """
        bsize = x.shape[0]
        # print(x.shape)
        # assert x.shape[-1] == x.shape[-2] == self.img_size
        y = x.clone()
        y = self.act(self.lin[0](y))
        pool_ind = 0
        x_scaled = x.clone()
        for i, (core, res) in enumerate(zip(self.lin[1:-2], self.res[:-1])):
            y = self.act(core(y) + res(x_scaled))
            if i in self.pool_inds:
                x_scaled = self.pool[pool_ind](x_scaled)
                y = self.pool[pool_ind](y)
                pool_ind += 1
                # print(y.shape)

        y = self.lin[-2](y) + self.res[-1](x_scaled)  # 1x1 if input is 64x64
        y = self.act(y)  # b, c, 1, 1

        y = self.lin[-1](y)  # b, 1, 1, 1

        # avg pooling
        # print(y.shape)
        y = torch.mean(y, dim=(2, 3))  # b, 1
        # print(y.shape)
        # strongly convex
        y = y + self.alpha * 64**2 * x.reshape(x.shape[0], -1).pow(2).mean(
            1, keepdim=True
        )
        # return shape: (batch, 1)
        return y * x.shape[-2] * x.shape[-1] / 64**2

    def init_weights(self, mean, std):
        print("init weights")
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.normal_(mean, std).exp_()

    # this clips the weights to be non-negative to preserve convexity
    def wclip(self):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.clamp_(0)

    def forward(self, x):
        """
        Grad of convex potential.
        Args:
            x: (b, c, w, h), normalized to [-1, 1]
        Note:
            The default input size is 64x64. For other sizes, the network extrapolates.
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.scalar(x)
            grad = torch.autograd.grad(
                y.sum(), x, retain_graph=True, create_graph=True
            )[0]
        return grad
