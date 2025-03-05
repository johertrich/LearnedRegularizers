"""
Created on Wed Mar 5 2025

@author: ZhenghanFang
"""

import torch
from deepinv.optim import Prior

from .lpn_original import LPN
from .prior import evaluate_prior
from .invert_model import invert


class LPNPrior(Prior):
    def __init__(
        self,
        in_dim=3,
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
