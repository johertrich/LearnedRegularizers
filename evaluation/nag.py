import torch
import numpy as np


def reconstruct_NAG(
    y,
    physics,
    data_fidelity,
    regularizer,
    lmbd,
    NAG_step_size,
    NAG_max_iter,
    NAG_tol,
    detach_grads=False,
    verbose=False,
):
    # run Nesterov Accelerated Gradient

    x = physics.A_dagger(y)
    z = x.clone()
    t = 1
    res = NAG_tol + 1
    for step in range(NAG_max_iter):
        x_old = torch.clone(x)
        grad = data_fidelity.grad(x, y, physics) + lmbd * regularizer.grad(x)
        if detach_grads:
            grad = grad.detach()
        x = z - NAG_step_size * grad
        x = x.clamp(0, 1)
        t_old = t
        t = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
        z = x + (t_old - 1) / t * (x - x_old)
        if step > 0:
            res_vec = torch.sqrt(
                torch.sum((x - x_old).view(x.shape[0], -1) ** 2)
                / torch.sum(x.view(x.shape[0], -1) ** 2)
            )
            res = torch.max(res_vec)
        if res < NAG_tol:
            if verbose:
                print(f"Converged in iter {step}, tol {res.item():.6f}")
            break
    if verbose and res >= NAG_tol:
        print(f"max iter reached, tol {res.item():.6f}")
    return x
