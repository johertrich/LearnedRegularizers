import numpy as np
from torch.utils.data import DataLoader
from .nag import reconstruct_NAG
import torch
from deepinv.loss.metric import PSNR


def evaluate(
    physics,
    data_fidelity,
    dataset,
    regularizer,
    lmbd,
    NAG_step_size,
    NAG_max_iter,
    NAG_tol,
    only_first=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
):

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    psnr = PSNR()

    regularizer.eval()
    for p in regularizer.parameters():
        p.requires_grad_(False)

    def reconstruct(y):
        # run Nesterov Accelerated Gradient

        x = physics.A_dagger(y)
        z = x.clone()
        t = 1
        res = NAG_tol + 1
        for step in range(NAG_max_iter):
            x_old = torch.clone(x)
            grad = (
                data_fidelity.grad(x, y, physics) + lmbd * regularizer.grad(x).detach()
            )
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
                break
        return x

    ## Evaluate on the test set

    psnrs = []
    for i, x in enumerate(dataloader):
        if device == "mps":
            # mps does not support float64
            x = x.to(torch.float32).to(device)
        else:
            x = x.to(device).to(torch.float)
        y = physics(x)
        recon = reconstruct_NAG(
            y,
            physics,
            data_fidelity,
            regularizer,
            lmbd,
            NAG_step_size,
            NAG_max_iter,
            NAG_tol,
            detach_grads=True,
            verbose=verbose,
        )
        psnrs.append(psnr(recon, x).squeeze().item())
        if i == 0:
            y_out = y
            x_out = x
            recon_out = recon
        if only_first:
            break
    mean_psnr = np.mean(psnrs)
    print("Mean PSNR over the test set: {0:.2f}".format(mean_psnr))
    return mean_psnr, x_out, y_out, recon_out
