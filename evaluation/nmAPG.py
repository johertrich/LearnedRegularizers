import torch
import numpy as np
from typing import Callable

# Implements Algorithm 4 from the supplementary material of
#
# Huan Li, Zhouchen Lin
# Accelerated Proximal Gradient Methods for Nonconvex Programming
# NeurIPS 2015
#
# [1] https://papers.nips.cc/paper_files/paper/2015/hash/f7664060cc52bc6f3d620bcedc94a4b6-Abstract.html
# [2] https://papers.nips.cc/paper_files/paper/2015/file/f7664060cc52bc6f3d620bcedc94a4b6-Supplemental.zip


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
    """
    Algorithm 4: nonmonotone APG with line search

    Solve for a given y: min_x f(x, y)

    In the notation of the paper F(x) = f(x, y).
    """

    # initialize variables
    x = x0.clone()  # Noation of the paper: x1
    x_old = x.clone()  # x0
    z = x0.clone()  # z1
    t = 1.0  # t1
    t_old = 0.0  # t0
    q = 1.0  # q1
    c = f(x, y)  # c1
    L = torch.full((x.shape[0], 1, 1, 1), L_init, dtype=torch.float32, device=x.device)
    L_old = L.clone()
    res = (tol + 1) * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    idx = torch.arange(0, x.shape[0], device=x.device)
    grad = torch.zeros_like(x)  # nabla F(x)
    x_bar = torch.zeros_like(x)
    x_bar_old = x_bar.clone()
    grad_old = grad.clone()

    # Main loop
    for i in range(max_iter):
        x_bar[idx] = (
            x[idx]
            + t_old / t * (z[idx] - x[idx])
            + (t_old - 1) / t * (x[idx] - x_old[idx])
        )  # Eq 148, x_bar = yk
        x_old.copy_(x)
        energy, grad[idx] = f(x_bar[idx], y[idx]), nabla(x_bar[idx], y[idx])

        # NOT CHECKED YET
        if i > 0:
            dx = grad[idx] - grad_old[idx]  # r in the paper
            s = (dx * dx).sum((1, 2, 3), keepdim=True)  # r^Tr
            L[idx] = torch.clip(
                s
                / (dx * (x_bar[idx] - x_bar_old[idx])).sum(
                    (1, 2, 3), keepdim=True
                ),  # alpha_y = <s,r>/<r,r> in paper, Eq 150
                min=1.0,
                max=None,
            )  # Why is there a lower bound of 1?

        # Repeat Eq 151 and 152
        for ii in range(150):
            z[idx] = x_bar[idx] - grad[idx] / L[idx]  # Eq 151, 1/L = alpha_y
            dx = z[idx] - x_bar[idx]
            bound = torch.max(
                energy[:, None, None, None], c[idx, None, None, None]
            ) - delta * (dx * dx).sum((1, 2, 3), keepdim=True)

            if torch.all((energy_new := f(z[idx], y[idx])) <= bound.view(-1)):
                break
            L[idx] = torch.where(
                energy_new[:, None, None, None] <= bound, L[idx], L[idx] / rho
            )  # the reduction with rho is a bit different in the paper as it reduces it also when bound is successful. Not sure this is meant though

        # If for Eq 153-158
        idx2 = (
            (energy_new[:] >= (c[idx] - delta * (dx * dx).sum((1, 2, 3))))
            .nonzero()
            .view(-1)
        )
        if idx2.nelement() > 0:
            gradx = nabla(x[idx[idx2]], y[idx[idx2]])  # nabla f(xk)

            if i > 0:
                dx = gradx - grad_old[idx[idx2]]
                s = (dx * dx).sum((1, 2, 3), keepdim=True)
                L[idx[idx2]] = torch.clip(
                    s
                    / (dx * (x[idx[idx2]] - x_bar_old[idx[idx2]])).sum(
                        (1, 2, 3), keepdim=True
                    ),
                    min=1.0,
                    max=None,
                )
            L_old.copy_(L)
            for ii in range(1500):
                v = x[idx[idx2]] - gradx / L[idx[idx2]]
                dx = v - x[idx[idx2]]
                bound = c[idx[idx2], None, None, None] - delta * (dx * dx).sum(
                    (1, 2, 3), keepdim=True
                )
                if torch.all(
                    (energy_new2 := f(v, y[idx[idx2]])) <= bound.view(-1) * (1 + 1e-4)
                ):
                    break
                L[idx[idx2]] = torch.where(
                    energy_new2[:, None, None, None] <= bound,
                    L[idx[idx2]],
                    L[idx[idx2]] / rho,
                )
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
        idx = condition.nonzero().view(-1)  # Update which data to still iterate on

        if torch.max(res) < tol:
            if verbose:
                print(f"Converged in iter {i}, tol {torch.max(res).item():.6f}")
            break
        t_old = t
        t = (np.sqrt(4.0 * t_old**2 + 1.0) + 1.0) / 2.0  # Eq 159
        q_old = q
        q = eta * q + 1.0  # Eq 160
        c[idx] = (eta * q_old * c[idx] + f(x[idx], y[idx])) / q  # Eq 161
        x_bar_old.copy_(x_bar)
        grad_old.copy_(grad)
    if verbose and (torch.max(res) >= tol):
        print(f"max iter reached, tol {torch.max(res).item():.6f}")
    if return_L:
        return x, L
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
    """wrapper for nmAPG"""

    if x_init is not None:
        # User-defined initialization or warm start
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y)

    def energy(val, y_in):
        fun = data_fidelity(val, y_in, physics) + lamda * regularizer.g(val)
        if detach_grads:
            fun = fun.detach()
        return fun.reshape(-1)

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
