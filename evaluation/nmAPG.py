import torch
import numpy as np
from typing import Callable
import inspect

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
    f_and_nabla: Callable[[torch.Tensor], [torch.Tensor]],
    max_iter: int = 200,
    L_init: float = 1,
    tol: float = 1e-4,
    rho: float = 0.9,
    delta: float = 0.1,
    eta: float = 0.8,
    verbose: bool = False,
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
        assert not torch.any(
            torch.isnan(x)
        ), "Numerical errors! Some values became NaN!"
        x_bar[idx] = (
            x[idx]
            + t_old / t * (z[idx] - x[idx])
            + (t_old - 1) / t * (x[idx] - x_old[idx])
        )  # Eq 148, x_bar = yk
        x_old.copy_(x)
        energy, grad[idx] = f_and_nabla(x_bar[idx], y[idx])

        # Lipschitz Update (Barzilai-Borwein style step)
        if i > 0:
            dx = grad[idx] - grad_old[idx]  # r in the paper
            s = (dx * dx).sum((1, 2, 3), keepdim=True)  # r^Tr
            L[idx] = torch.clip(
                s
                / (dx * (x_bar[idx] - x_bar_old[idx]))
                .sum((1, 2, 3), keepdim=True)
                .abs()
                .clip(min=1e-12, max=None),  # alpha_y = <s,r>/<r,r> in paper, Eq 150
                min=1.0,
                max=None,
            )  # clips for stability --> on a long term we can adjust min-clip based on the spectral norm of physics.A
        # line search on z (Eq 151 and 152)
        idx_search = idx
        idx_sub = torch.arange(0, idx.shape[0], device=x.device)
        energy_new = energy.clone()
        dx = z[idx] - x_bar[idx]
        for ii in range(150):
            z[idx_search] = (
                x_bar[idx_search] - grad[idx_search] / L[idx_search]
            )  # Eq 151, 1/L = alpha_y
            dx[idx_sub] = z[idx_search] - x_bar[idx_search]
            bound = torch.max(
                energy[idx_sub, None, None, None], c[idx_search, None, None, None]
            ) - delta * (dx[idx_sub] * dx[idx_sub]).sum((1, 2, 3), keepdim=True)

            if torch.all(
                (energy_new_ := f(z[idx_search], y[idx_search])) <= bound.view(-1)
            ):
                energy_new[idx_sub] = energy_new_
                break
    
            energy_new[idx_sub] = energy_new_
            idx_sub = idx_sub[energy_new_ > bound.view(-1)]
            idx_search = idx[idx_sub]
            L[idx_search] = L[idx_search] / rho

        # If for Eq 153-158
        idx2 = (
            (energy_new[:] >= (c[idx] - delta * (dx * dx).sum((1, 2, 3))))
            .nonzero()
            .view(-1)
        )
        if idx2.nelement() > 0:
            idx_idx2 = idx[idx2]
            gradx = nabla(x[idx_idx2], y[idx_idx2])  # nabla f(xk)

            if i > 0:
                dx = gradx - grad_old[idx_idx2]
                s = (dx * dx).sum((1, 2, 3), keepdim=True)
                L[idx_idx2] = torch.clip(
                    s
                    / (dx * (x[idx_idx2] - x_bar_old[idx_idx2]))
                    .sum((1, 2, 3), keepdim=True)
                    .abs()
                    .clip(min=1e-12, max=None),
                    min=1.0,
                    max=None,
                )
            L_old.copy_(L)

            # Line search on v
            for ii in range(150):
                v = x[idx_idx2] - gradx / L[idx_idx2]
                dx = v - x[idx_idx2]
                bound = c[idx_idx2, None, None, None] - delta * (dx * dx).sum(
                    (1, 2, 3), keepdim=True
                )
                if torch.all(
                    (energy_new2 := f(v, y[idx_idx2])) <= bound.view(-1) * (1 + 1e-4)
                ):
                    break
                L[idx_idx2] = torch.where(
                    energy_new2[:, None, None, None] <= bound,
                    L[idx_idx2],
                    L[idx_idx2] / rho,
                )
            x[idx] = z[idx]
            idx3 = (energy_new2 <= energy_new[idx2]).nonzero().view(-1)
            tmp = idx_idx2[idx3]
            x[tmp] = v[idx3]
        else:
            x[idx] = z[idx]

        if i > 0:
            res[idx] = torch.norm(x[idx] - x_old[idx], p=2, dim=(1, 2, 3)) / torch.norm(
                x[idx], p=2, dim=(1, 2, 3)
            )
        assert not torch.any(
            torch.isnan(res)
        ), "Numerical errors! Some values became NaN!"
        condition = res >= tol
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
    converged = res < tol
    return x, L, i, converged


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
    return_stats=False,
):
    """wrapper for nmAPG"""

    if x_init is not None:
        # User-defined initialization or warm start
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y)

    def energy(val, y_in):
        with torch.no_grad():
            fun = data_fidelity(val, y_in, physics) + lamda * regularizer.g(val)
        if detach_grads:
            fun = fun.detach()
        return fun.reshape(-1)

    def energy_grad(val, y_in):
        grad = data_fidelity.grad(val, y_in, physics) + lamda * regularizer.grad(val)
        if detach_grads:
            grad = grad.detach()
        return grad

    # check if energy can be accessed during grad evaluation
    signature = inspect.signature(regularizer.grad)
    argument_names = [param.name for param in signature.parameters.values()]
    if "get_energy" in argument_names:

        def energy_and_grad(val, y_in):
            fun, grad = regularizer.grad(val, get_energy=True)
            fun = data_fidelity(val, y_in, physics) + lamda * fun
            grad = data_fidelity.grad(val, y_in, physics) + lamda * grad
            if detach_grads:
                fun = fun.detach()
                grad = grad.detach()
            return fun.reshape(-1), grad

    else:
        energy_and_grad = lambda val, y_in: (energy(val, y_in), energy_grad(val, y_in))

    # example energies
    rec, L, steps, converged = nmAPG(
        x0=x,
        y=y,
        max_iter=max_iter,
        f=energy,
        nabla=energy_grad,
        f_and_nabla=energy_and_grad,
        L_init=1 / step_size,
        tol=tol,
        verbose=verbose,
    )
    stats = dict(L=L.detach(), steps=steps, converged=converged)
    if return_stats:
        return rec, stats
    return rec
