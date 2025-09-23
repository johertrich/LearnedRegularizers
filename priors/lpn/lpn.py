"""
Created on Wed Mar 5 2025

@author: ZhenghanFang
"""

import torch
from deepinv.optim import Prior
from torch import nn

from .prior import evaluate_prior


class LPNPrior(Prior):
    """Wrapper of LPN as a deepinv.optim.Prior"""

    def __init__(
        self,
        in_dim=1,
        hidden=256,
        beta=100.0,
        alpha=1e-6,
        stride_size=32,
        pretrained=None,
        clip=False,
    ):
        """
        Args:
            in_dim: int, input dimension
            hidden: int, hidden channel size
            beta: float, beta in softplus
            alpha: float, strongly-convex parameter
        """
        super().__init__()
        self.lpn = LPN(in_dim, hidden, beta, alpha, stride_size=stride_size)
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location="cpu"))
        self.clip = clip

    def g(self, x: torch.Tensor, inv_alg="cvx_cg", **kwargs) -> torch.Tensor:
        """
        Regularizer value.
        Args:
            x: torch.Tensor, shape (B, *), normalized to [0, 1]

        Returns:
            torch.Tensor, shape (B, ), prior at each sample
        """
        x = x * 2 - 1  # convert to [-1, 1]
        x_np = x.detach().cpu().numpy()
        prior = evaluate_prior(x_np, self.lpn, inv_alg=inv_alg, **kwargs)["p"]
        return torch.tensor(prior, device=x.device)

    def prox(self, x, *args, **kwargs):
        """
        Proximal operator of the regularizer.
        Args:
            x: torch.Tensor, shape (B, *), normalized to [0, 1]
        """
        x = x * 2 - 1  # convert to [-1, 1]
        out = self.lpn(x)
        # convert back to [0, 1]
        out = (out + 1) / 2

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


def pad_for_exact_prox(x, patch_size, stride_size):
    """Pad x so that each pixel is covered by the same number of patches.
    Args:
        x: (B, C, H, W), torch tensor
        patch_size: int
        stride_size: int
    Returns:
        x: (B, C, H', W'), padded torch tensor
    """
    # First, pad so that image size is divisible by stride size
    w, h = x.shape[-1], x.shape[-2]
    pad_w = 0 if w % stride_size == 0 else stride_size - (w % stride_size)
    pad_h = 0 if h % stride_size == 0 else stride_size - (h % stride_size)
    pad = (0, pad_w, 0, pad_h)
    # Next, pad patch_size - stride_size on all sides
    pad = (
        pad[0] + (patch_size - stride_size),
        pad[1] + (patch_size - stride_size),
        pad[2] + (patch_size - stride_size),
        pad[3] + (patch_size - stride_size),
    )

    x = torch.nn.functional.pad(x, pad)
    assert x.shape[-1] % stride_size == 0 and x.shape[-2] % stride_size == 0
    return x, pad


def get_patch_coors_1d(size, p, s):
    out = list(range(0, size - p + 1, s))
    if out[-1] != size - p:
        out.append(size - p)
    return out


def get_patch_coors_2d(h, w, p, s):
    i_list = get_patch_coors_1d(h, p, s)
    j_list = get_patch_coors_1d(w, p, s)
    return [(i, j) for i in i_list for j in j_list]


class LPN(nn.Module):
    """Learned proximal networks.
    Modified from https://github.com/ZhenghanFang/learned-proximal-networks/blob/main/lpn/networks/lpn_128.py

    Default image size is 64x64.
    This version enables applying to larger images by patches while ensuring
    the computation on the whole image is still a proximal operator.
    """

    def __init__(self, in_dim, hidden, beta, alpha, stride_size=32):
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
        self.stride_size = stride_size  # stride size for larger images

        # init weights
        self.init_weights(-10, 0.1)

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

    def _scalar(self, x):
        """Base convex scalar function. ICNN.
        Args:
            x: (b, c, 64, 64)
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

    def _scalar_larger_size(self, x):
        """Convex scalar potential for larger input."""
        patch_size = self.img_size
        stride_size = self.stride_size
        x_pad, pad = pad_for_exact_prox(x, patch_size, stride_size)
        patch_coors = get_patch_coors_2d(
            x_pad.shape[-2], x_pad.shape[-1], patch_size, stride_size
        )
        out = 0
        for idx, (i, j) in enumerate(patch_coors):
            patch = x_pad[:, :, i : i + patch_size, j : j + patch_size]
            out_patch = self._scalar(patch)
            out = out + out_patch
        return out / ((patch_size / stride_size) ** 2)

    def _prox(self, x):
        """
        Proximal operator via gradient of scalar function.
        Args:
            x: (b, c, 64, 64)
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self._scalar(x)
            grad = torch.autograd.grad(
                y.sum(), x, retain_graph=True, create_graph=True
            )[0]
        return grad

    def _prox_larger_size(self, x, mode="patch_avg"):
        """Proximal operator for larger input."""
        if mode == "patch_avg":
            # patch-wise computation + average (less gpu memory)
            patch_size = self.img_size
            stride_size = self.stride_size
            x_pad, pad = pad_for_exact_prox(x, patch_size, stride_size)
            patch_coors = get_patch_coors_2d(
                x_pad.shape[-2], x_pad.shape[-1], patch_size, stride_size
            )
            out = torch.zeros_like(x_pad)
            count = torch.zeros_like(x_pad)
            for idx, (i, j) in enumerate(patch_coors):
                patch = x_pad[:, :, i : i + patch_size, j : j + patch_size]
                out_patch = self._prox(patch)
                out[:, :, i : i + patch_size, j : j + patch_size] += out_patch
                count[:, :, i : i + patch_size, j : j + patch_size] += 1
            out = out[
                :, :, pad[2] : out.shape[2] - pad[3], pad[0] : out.shape[3] - pad[1]
            ]
            count = count[
                :, :, pad[2] : count.shape[2] - pad[3], pad[0] : count.shape[3] - pad[1]
            ]
            assert torch.all(
                count == (patch_size / stride_size) ** 2
            ), f"Each pixel should be covered by {(patch_size / stride_size) ** 2} patches."
            out = out / count
            return out

        elif mode == "grad":
            # direct gradient computation (more gpu memory)
            with torch.enable_grad():
                x.requires_grad_(True)
                y = self._scalar_larger_size(x)
                grad = torch.autograd.grad(
                    y.sum(), x, retain_graph=True, create_graph=True
                )[0]
            return grad

        else:
            raise NotImplementedError(f"Unknown mode {mode}")

    def scalar(self, x):
        """
        Convex scalar function.
        Args:
            x: (b, c, w, h)
        """
        if x.shape[-1] == x.shape[-2] == self.img_size:
            return self._scalar(x)
        else:
            return self._scalar_larger_size(x)

    def forward(self, x):
        """
        Proximal operator.
        Args:
            x: (b, c, w, h)
        """
        if x.shape[-1] == x.shape[-2] == self.img_size:
            grad = self._prox(x)
        else:
            grad = self._prox_larger_size(x)

        return grad
