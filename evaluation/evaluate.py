import numpy as np
from torch.utils.data import DataLoader
from .nag_ls import reconstruct_NAG_LS
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

    ## Evaluate on the test set
    psnrs = []
    for i, x in enumerate(dataloader):
        if device == "mps":
            # mps does not support float64
            x = x.to(torch.float32).to(device)
        else:
            x = x.to(device).to(torch.float)
        y = physics(x)
        recon = reconstruct_NAG_LS(
            y,
            physics,
            data_fidelity,
            regularizer,
            lmbd,
            NAG_step_size,
            NAG_max_iter,
            NAG_tol,
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
