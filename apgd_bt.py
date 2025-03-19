import torch
from typing import Callable


def apgd(
    x0: torch.Tensor,
    f: Callable[[torch.Tensor], torch.Tensor],
    nabla: Callable[[torch.Tensor], torch.Tensor],
    prox: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    callback: Callable[[torch.Tensor], None] = lambda _: None,
    max_iter: int = 200,
    L_init: float = 1,
):
    x = x0.clone()
    x_old = x.clone()
    L = torch.full((x.shape[0], 1, 1, 1), L_init, dtype=torch.float32, device=x.device)
    for i in range(max_iter):
        beta = i / (i + 3)
        x_bar = x + beta * (x - x_old)
        x_old = x.clone()
        energy, grad = f(x_bar), nabla(x_bar)
        for _ in range(50):
            x = prox(x_bar - grad / L, 1 / L)
            dx = x - x_bar
            bound = (
                energy
                + (grad * dx).sum((1, 2, 3), keepdim=True)
                + L * (dx * dx).sum((1, 2, 3), keepdim=True) / 2
            )
            if torch.all((energy_new := f(x)) <= bound):
                break
            L = torch.where(energy_new <= bound, L, 2 * L)
        L /= 1.5
        callback(x)
    return x


def reconstruct_apgd_bt(
    y,
    physics,
    data_fidelity,
    regularizer,
    lamda,
    max_iter,
    x_init=None,
    detach_grads=True,
):
    if x_init is not None:
        # User-defined initialization or warm start
        x = torch.clone(x_init).detach().requires_grad_(True)
    else:
        x = physics.A_dagger(y)

    def energy(val):
        return data_fidelity(val, y, physics) + lamda * regularizer.g(val)

    def energy_grad(val):
        grad = data_fidelity.grad(val, y, physics) + lamda * regularizer.grad(val)
        if detach_grads:
            grad = grad.detach()
        return grad

    if x_init is not None:
        # User-defined initialization or warm start
        x = torch.clone(x_init).detach().requires_grad_(True)
    else:
        x = physics.A_dagger(y)

    # example energies
    # energies = torch.empty((max_iter,))
    # i = 0
    # def cb(x):
    #     nonlocal i
    #     energies[i] = energy(x)
    #     i += 1
    def cb(_):
        return None

    rec = apgd(
        x0=x,
        max_iter=max_iter,
        f=energy,
        nabla=energy_grad,
        prox=lambda x, _: x,
        callback=cb,
    )
    # torch.save(energies, 'energies_apgd.pt')
    return rec
