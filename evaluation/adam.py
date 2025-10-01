"""
This file provides a function for using Adam to minimize a variational problem 
data_fideltiy(physics(x),y)+lmbd*regularizer(x)
with Adam and a cosine scheduler.
"""

import torch
from tqdm import tqdm
import torch.nn.functional as F


def reconstruct_adam(
    y,  # observation in the variational problem
    physics,  # deepinv physics object defining the forward operator and the noise model
    data_fidelity,  # deepinv data fidelity object defining the data fidelity term
    regularizer,  # used regularizer
    lamda,  # regularization parameter
    step_size,  # initial step size
    max_iter,  # maximum number of iterations
    tol,  # tolerance for the stopping criterion (relative residual)
    x_init=None,  # initialization (None for physics.A_dagger(y))
    detach_grads=True,  # detach gradients after each step (should be set to True to prevent memory leakage)
    verbose=False,  # set to True for some debug logs
    return_stats=False,  # set to True to return some stats (number of used steps etc) in addition to the reconstruction
):
    """wrapper for adam"""

    if x_init is not None:
        # User-defined initialization or warm start
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y)

    def energy(val, y_in):
        fun = data_fidelity(val, y_in, physics) + lamda * regularizer.g(val)

        return fun.reshape(-1)

    x.requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=step_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iter, eta_min=step_size / 10.0
    )

    for i in range(max_iter):
        x_old = x.detach().clone()
        optimizer.zero_grad()
        loss = energy(x, y)
        loss.backward()
        optimizer.step()

        scheduler.step()
        with torch.no_grad():
            x.data.clamp_(min=0)
        residual = torch.norm(x - x_old) / torch.norm(x_old)
        if residual < tol:
            if verbose:
                print(
                    "Converged after {} steps with residual {}".format(i + 1, residual)
                )
            break
    stats = dict(steps=i + 1)

    del optimizer
    del scheduler

    rec = x.detach()
    if return_stats:
        return rec, stats
    return rec
