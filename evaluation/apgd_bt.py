import torch
from typing import Callable


def apgd(
    x0: torch.Tensor,
    y: torch.Tensor,
    f: Callable[[torch.Tensor], torch.Tensor],
    nabla: Callable[[torch.Tensor], torch.Tensor],
    max_iter: int = 200,
    L_init: float = 1,
    tol: float = 1e-4,
    rho: float = 0.9,
    delta: float = 0.9,
    verbose: bool = False,
):
    x = x0.clone()
    x_old = x.clone()
    L = torch.full((x.shape[0], 1, 1, 1), L_init, dtype=torch.float32, device=x.device)
    res = (tol + 1) * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    idx = torch.arange(0, x.shape[0], device=x.device)
    for i in range(max_iter):
        beta = i / (i + 3)
        x_bar = x[idx] + beta * (x[idx] - x_old[idx])
        x_old = x.clone()
        energy, grad = f(x_bar, y[idx]), nabla(x_bar, y[idx])
        for ii in range(50):
            x[idx] = x_bar - grad / L[idx]
            dx = x[idx] - x_bar
            bound = (
                energy[:, None, None, None]
                + (grad * dx).sum((1, 2, 3), keepdim=True)
                + L[idx] * (dx * dx).sum((1, 2, 3), keepdim=True) / 2
            )

            if torch.all((energy_new := f(x[idx], y[idx])) <= bound.view(-1)):
                break
            L[idx] = torch.where(
                energy_new[:, None, None, None] <= bound, L[idx], L[idx] / rho
            )
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

        L *= delta
    if verbose and (torch.max(res) >= tol):
        print(f"max iter reached, tol {torch.max(res).item():.6f}")
    return x


def reconstruct_apgd_bt(
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
    rec = apgd(
        x0=x,
        y=y,
        max_iter=max_iter,
        f=energy,
        nabla=energy_grad,
        L_init=1 / step_size,
        tol=tol,
        verbose=verbose,
    )
    return rec
