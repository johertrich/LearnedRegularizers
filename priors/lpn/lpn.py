"""
Created on Wed Mar 5 2025

@author: ZhenghanFang
"""

import torch
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
        stride_size=None,
        exact_prox=True,
    ):
        """
        Args:
            in_dim: int, input dimension
            hidden: int, hidden channel size
            beta: float, beta in softplus
            alpha: float, strongly-convex parameter
            model_name: str, select lpn model
            exact_prox: bool, ensure exact proximal operator when applying by patches
        """
        super().__init__()
        self.lpn = LPN(in_dim, hidden, beta, alpha)
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location="cpu"))
        self.clip = clip
        self.stride_size = stride_size
        self.exact_prox = exact_prox

    def g(self, x: torch.Tensor, inv_alg="cvx_cg", **kwargs) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, shape (B, *)

        Returns:
            torch.Tensor, shape (B, ), prior at each sample
        """
        x_np = x.detach().cpu().numpy()
        prior = evaluate_prior(x_np, self.lpn, inv_alg=inv_alg, **kwargs)["p"]
        return torch.tensor(prior, device=x.device)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the gradient of the regularizer at x by f^{-1}(x) - x, where
        f^{-1} is the inverse of LPN.

        Args:
            x: torch.Tensor, shape (B, *)

        Returns:
            torch.Tensor, shape (B, *)
        """
        x_np = x.detach().cpu().numpy()
        y = invert(x_np, self.lpn, inv_alg="cvx_cg")
        y = torch.tensor(y, device=x.device)
        return y - x

    def prox(self, x, *args, **kwargs):
        if x.shape[-1] == x.shape[-2] == self.lpn.img_size:
            """
            Compute prox directly via LPN.
            x: (B, C, H, W)
            """
            out = self.lpn(x)
        elif self.exact_prox:
            """
            Compute prox by patches. Ensure the computation on the whole image
            is a proximal operator by padding.
            x: (B, C, H, W)
            """
            patch_size = self.lpn.img_size
            stride_size = self.stride_size or self.lpn.img_size // 2
            assert patch_size % stride_size == 0, "stride should divide patch size"

            # Pad so that image size is divisible by stride size
            w, h = x.shape[-1], x.shape[-2]
            pad_w = 0 if w % stride_size == 0 else stride_size - (w % stride_size)
            pad_h = 0 if h % stride_size == 0 else stride_size - (h % stride_size)
            orig_shape = x.shape
            x = torch.nn.functional.pad(
                x,
                (0, pad_w, 0, pad_h),
                mode="constant",
            )

            print(x.shape)

            # Call the patch-wise processing function with pad_for_exact_prox=True
            # to ensure each pixel is covered by the same number of patches,
            # which ensures exact proximal operator on the whole image
            out = apply_func_to_patches(
                x, self.lpn, patch_size, stride_size, pad_for_exact_prox=True
            )
            out = out[:, :, : orig_shape[2], : orig_shape[3]]
        else:
            """
            Compute prox by patches.
            x: (B, C, H, W)
            """
            patch_size = self.lpn.img_size
            stride_size = self.stride_size or self.lpn.img_size // 2
            out = apply_func_to_patches(
                x, self.lpn, patch_size, stride_size, pad_for_exact_prox=False
            )

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


def apply_func_to_patches(
    x: torch.Tensor,
    func,
    patch_size,
    stride_size,
    pad_for_exact_prox=True,
) -> torch.Tensor:
    """Apply a func to an image by patches
    Inputs:
        x: image to be processed, shape: (B, C, H, W)
        func: callable
        patch_size: size of patch
        stride_size: stride for patch
    Outputs:
        xhat: processed image, shape: (B, C, H, W)
    """
    if pad_for_exact_prox:
        assert patch_size % 2 == 0
        x = torch.nn.functional.pad(
            x,
            (patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2),
            mode="constant",
        )  # pad to ensure a proximal operator

    # Compute coordinates of patches
    def get_coors_1d(size, p, s):
        out = list(range(0, size - p + 1, s))
        # print(out[-1], size - p)
        if out[-1] != size - p:
            out.append(size - p)
        return out

    i_list = get_coors_1d(x.shape[2], patch_size, stride_size)
    j_list = get_coors_1d(x.shape[3], patch_size, stride_size)
    # print(i_list, j_list)

    xhat = torch.zeros_like(x)
    count = torch.zeros_like(x)
    for i in i_list:
        for j in j_list:
            with torch.no_grad():
                xhat[:, :, i : i + patch_size, j : j + patch_size] += func(
                    x[:, :, i : i + patch_size, j : j + patch_size]
                )
                count[:, :, i : i + patch_size, j : j + patch_size] += 1

    if pad_for_exact_prox:
        xhat = xhat[
            :, :, patch_size // 2 : -patch_size // 2, patch_size // 2 : -patch_size // 2
        ]
        count = count[
            :, :, patch_size // 2 : -patch_size // 2, patch_size // 2 : -patch_size // 2
        ]
        # print(count.min(), count.max())
        # print(stride_size)
        assert torch.all(
            count == count.flatten()[0]
        )  # each pixel should be covered by the same number of patches
        # print("Each pixel is covered by", count.flatten()[0].item(), "patches")
    xhat = xhat / count
    return xhat


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
