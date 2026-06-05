"""Adam solver for the variational problem.

Pure optimization routine: minimise a *differentiable* objective ``energy(x, y)``
with ``torch.optim.Adam`` and a cosine-annealing schedule, projecting the iterate
to the non-negative orthant after every step. The reconstruction wrapper that
builds the inverse-problem objective lives in ``evaluation/reconstruct.py``.

Unlike :func:`nmAPG`, Adam relies on autograd, so ``energy`` must build a graph
(i.e. it must *not* be wrapped in ``torch.no_grad``). It is called as
``energy(x, y) -> (B,)`` (or a scalar); the per-sample energies are summed before
backpropagation, which is exact for the independent per-sample problems handled
here.
"""

import torch
from typing import Callable


def adam(
    x0: torch.Tensor,  # initial point x0
    y: torch.Tensor,  # additional parameter y of the objective function
    energy: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ],  # differentiable objective
    step_size: float,  # initial step size (Adam learning rate)
    max_iter: int,  # maximum number of iterations
    tol: float,  # tolerance for the stopping criterion (relative residual)
    verbose: bool = False,  # set to True for some debug prints
):
    """Minimise ``energy(x, y)`` with Adam + cosine annealing."""

    x = x0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=step_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iter, eta_min=step_size / 10.0
    )

    converged = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    x_prev = x.detach().clone()
    i = 0
    for i in range(max_iter):
        x_prev.copy_(x.detach())
        optimizer.zero_grad()
        loss = energy(x, y).reshape(-1).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            x.data.clamp_(min=0)
            residual = (x - x_prev).norm() / x.norm().clamp_min(1e-12)
        if residual < tol:
            if verbose:
                print(
                    "Converged after {} steps with residual {}".format(i + 1, residual)
                )
            converged[:] = True
            break

    del optimizer
    del scheduler

    return x.detach(), i + 1, converged
