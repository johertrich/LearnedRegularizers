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
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=2, padding=1),  # 32
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=1, padding=1),  # 32
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=2, padding=1),  # 16
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=1, padding=1),  # 16
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=2, padding=1),  # 8
                nn.Conv2d(hidden, 64, 8, bias=False, stride=1, padding=0),  # 1
                nn.Conv2d(64, 1, 1),
            ]
        )

        self.res = nn.ModuleList(
            [
                nn.Conv2d(in_dim, hidden, 3, stride=2, padding=1),  # 32
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1),  # 32
                nn.Conv2d(in_dim, hidden, 3, stride=2, padding=1),  # 16
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1),  # 16
                nn.Conv2d(in_dim, hidden, 3, stride=2, padding=1),  # 8
                nn.Conv2d(in_dim, 64, 8, stride=1, padding=0),  # 1
            ]
        )

        self.act = nn.Softplus(beta=beta)
        self.alpha = alpha

        self.img_size = 64  # expected image size

        # init weights
        self.init_weights(-10, 0.1)

    def _scalar(self, x):
        """Base scalar function. ICNN.
        Args:
            x: (b, c, 64, 64), normalized to [-1, 1]
        """
        bsize = x.shape[0]
        assert x.shape[-1] == x.shape[-2] == self.img_size
        image_size = x.shape[-1]
        y = x.clone()
        y = self.act(self.lin[0](y))
        size = [
            image_size,
            image_size // 2,
            image_size // 2,
            image_size // 4,
            image_size // 4,
            image_size // 8,
        ]
        for core, res, sz in zip(self.lin[1:-2], self.res[:-1], size[:-1]):
            x_scaled = nn.functional.interpolate(x, (sz, sz), mode="bilinear")
            y = self.act(core(y) + res(x_scaled))

        x_scaled = nn.functional.interpolate(x, (size[-1], size[-1]), mode="bilinear")
        y = self.lin[-2](y) + self.res[-1](x_scaled)  # 1x1 if input is 64x64
        y = self.act(y)  # b, c, 1, 1

        y = self.lin[-1](y)  # b, 1, 1, 1
        # print(y.shape)

        # ensure the input size is 64 for now
        assert y.shape[2] == y.shape[3] == 1
        # avg pooling
        y = torch.mean(y, dim=(2, 3))  # b, 1
        # print(y.shape)

        # strongly convex
        y = y + self.alpha * x.reshape(x.shape[0], -1).pow(2).sum(1, keepdim=True)

        # return shape: (batch, 1)
        return y

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
        Args:
            x: (b, c, 64, 64), normalized to [0, 1]
        """
        x = x * 2 - 1  # convert to [-1, 1]
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            x_ = x
            y = self._scalar(x_)
            grad = torch.autograd.grad(
                y.sum(), x_, retain_graph=True, create_graph=True
            )[0]

        # convert back to [0, 1]
        grad = (grad + 1) / 2
        return grad

    def scalar(self, x):
        """
        Scalar function. ICNN.
        Args:
            x: (b, c, 64, 64), normalized to [0, 1]
        """
        x = x * 2 - 1  # convert to [-1, 1]
        return self._scalar(x)
