import torch
import numpy as np
from typing import Callable

# Implements Algorithm 4 from the paper
#
# Huan Li, Zhouchen Lin
# Accelerated Proximal Gradient Methods for Nonconvex Programming
# NeurIPS 2015


def nmAPG(
    x0: torch.Tensor,
    y: torch.Tensor,
    f: Callable[[torch.Tensor], torch.Tensor],
    nabla: Callable[[torch.Tensor], torch.Tensor],
    max_iter: int = 200,
    L_init: float = 1,
    tol: float = 1e-4,
    rho: float = 0.9,
    delta: float = 0.1,
    eta: float = 0.8,
    verbose: bool = False,
    return_L: bool = False,
):
    x = x0.clone()
    x_old = x.clone()
    z = x0.clone()
    t = 1.0
    t_old = 0.0
    q = 1.0
    c = f(x, y)
    L = torch.full((x.shape[0], 1, 1, 1), L_init, dtype=torch.float32, device=x.device)
    L_out = L.clone()
    res = (tol + 1) * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    idx = torch.arange(0, x.shape[0], device=x.device)
    grad = torch.zeros_like(x)
    x_bar = torch.zeros_like(x)
    for i in range(max_iter):
        x_bar[idx] = (
            x[idx]
            + t_old / t * (z[idx] - x[idx])
            + (t_old - 1) / t * (x[idx] - x_old[idx])
        )
        x_old = x.clone()
        energy, grad[idx] = f(x_bar[idx], y[idx]), nabla(x_bar[idx], y[idx])
        if i > 0:
            dx = grad[idx] - grad_old[idx]
            s = (dx * dx).sum((1, 2, 3), keepdim=True)
            L = torch.clip(
                s / (dx * (x_bar[idx] - x_bar_old[idx])).sum((1, 2, 3), keepdim=True),
                min=1.0,
                max=None,
            )
            L_out[idx] = L
        for ii in range(50):
            z[idx] = x_bar[idx] - grad[idx] / L
            dx = z[idx] - x_bar[idx]
            bound = torch.max(
                energy[:, None, None, None], c[idx, None, None, None]
            ) - delta * (dx * dx).sum((1, 2, 3), keepdim=True)

            if torch.all((energy_new := f(z[idx], y[idx])) <= bound.view(-1)):
                break
            L = torch.where(energy_new[:, None, None, None] <= bound, L, L / rho)
            L_out[idx] = L
        idx2 = (
            (energy_new[:] >= (c[idx] - delta * (dx * dx).sum((1, 2, 3))))
            .nonzero()
            .view(-1)
        )
        if idx2.nelement() > 0:
            gradx = nabla(x[idx][idx2], y[idx][idx2])
            if i > 0:
                dx = gradx - grad_old[idx][idx2]
                s = (dx * dx).sum((1, 2, 3), keepdim=True)
                L = torch.clip(
                    s
                    / (dx * (x[idx][idx2] - x_bar_old[idx][idx2])).sum(
                        (1, 2, 3), keepdim=True
                    ),
                    min=1.0,
                    max=None,
                )
                L_out[idx][idx2] = L
            for ii in range(50):
                v = x[idx][idx2] - gradx / L
                dx = v - x[idx][idx2]
                bound = c[idx2, None, None, None] - delta * (dx * dx).sum(
                    (1, 2, 3), keepdim=True
                )
                if torch.all((energy_new2 := f(v, y[idx][idx2])) <= bound.view(-1)):
                    break
                L = torch.where(energy_new2[:, None, None, None] <= bound, L, L / rho)
                L_out[idx][idx2] = L
            x[idx] = z[idx]
            idx3 = (energy_new2 <= energy_new[idx2]).nonzero().view(-1)
            tmp = idx[idx2][idx3]
            x[tmp] = v[idx3]
        else:
            x[idx] = z[idx]
        if i > 0:
            res[idx] = torch.norm(x[idx] - x_old[idx], p=2, dim=(1, 2, 3)) / torch.norm(
                x[idx], p=2, dim=(1, 2, 3)
            )
        condition = res > tol
        idx = condition.nonzero().view(-1)

        if torch.max(res) < tol:
            if verbose:
                print(f"Converged in iter {i}, tol {torch.max(res).item():.6f}")
            break
        t_old = t
        t = (np.sqrt(4.0 * t_old**2 + 1.0) + 1.0) / 2.0
        q_old = q
        q = eta * q + 1.0
        c[idx] = (eta * q_old * c[idx] + f(x[idx], y[idx])) / q
        x_bar_old = x_bar.clone()
        grad_old = grad.clone()
    if verbose and (torch.max(res) >= tol):
        print(f"max iter reached, tol {torch.max(res).item():.6f}")
    if return_L:
        return x, L_out
    return x


def reconstruct_nmAPG(
    y,
    physics,
    data_fidelity,
    regularizer,
    lamda,
    step_size,
    max_iter,
    tol,
    x_init=None,
    detach_grads=True,
    verbose=False,
    return_L=False,
):
    if x_init is not None:
        # User-defined initialization or warm start
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y)

    def energy(val, y_in):
        fun = data_fidelity(val, y_in, physics) + lamda * regularizer.g(val)
        if detach_grads:
            fun = fun.detach()
        return fun

    def energy_grad(val, y_in):
        grad = data_fidelity.grad(val, y_in, physics) + lamda * regularizer.grad(val)
        if detach_grads:
            grad = grad.detach()
        return grad

    # example energies
    rec = nmAPG(
        x0=x,
        y=y,
        max_iter=max_iter,
        f=energy,
        nabla=energy_grad,
        L_init=1 / step_size,
        tol=tol,
        verbose=verbose,
        return_L=return_L,
    )
    return rec
