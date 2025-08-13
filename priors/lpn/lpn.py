"""
Created on Wed Mar 5 2025

@author: ZhenghanFang
"""

import torch
from deepinv.optim import Prior


from .prior import evaluate_prior
from .invert_model import invert
from .lpn_64_neg1 import LPN


class LPNPrior(Prior):
    def __init__(
        self,
        in_dim=1,
        hidden=256,
        beta=100.0,
        alpha=1e-6,
        pretrained=None,
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

    def g(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, shape (B, *)

        Returns:
            torch.Tensor, shape (B, ), prior at each sample
        """
        x_np = x.detach().cpu().numpy()
        prior = evaluate_prior(x_np, self.lpn, inv_alg="cvx_cg")["p"]
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
            return self.lpn(x)
        else:
            """
            Compute prox by patches.
            x: (B, C, H, W)
            """
            patch_size = self.lpn.img_size
            stride_size = self.lpn.img_size // 2
            return apply_func_to_patches(x, self.lpn, patch_size, stride_size)

    def forward(self, x):
        return self.prox(x)


def apply_func_to_patches(
    x: torch.Tensor, func, patch_size, stride_size
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

    # Compute coordinates of patches
    def get_coors_1d(size, p, s):
        out = list(range(0, size - p + 1, s))
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
    xhat = xhat / count
    return xhat
