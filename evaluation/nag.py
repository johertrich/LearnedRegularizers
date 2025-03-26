import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt 
from deepinv.loss.metric import PSNR

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
    x_init=None,
    x_gt = None
):
    # run Nesterov Accelerated Gradient
    if x_init is not None:
        # User-defined initialization or warm start
        x = torch.clone(x_init).detach().requires_grad_(True)
    else:
        x = physics.A_dagger(y)
    z = torch.clone(x)

    t = 1
    res = NAG_tol + 1
    psnr = PSNR()
    for step in tqdm(range(NAG_max_iter), disable=not verbose):
        x_old = torch.clone(x)
        
        reg_grad = regularizer.grad(x)
        data_fidelity_grad = data_fidelity.grad(x, y, physics)

        #print(torch.linalg.norm(reg_grad), torch.linalg.norm(data_fidelity_grad))

        grad = data_fidelity_grad + lmbd * reg_grad
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
        
        if x_gt is not None:
            print("PSNR: ", psnr(x, x_gt).squeeze().item())
    if verbose and res >= NAG_tol:
        print(f"max iter reached, tol {res.item():.6f}")
    return x
