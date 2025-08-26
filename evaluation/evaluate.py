import numpy as np
from torch.utils.data import DataLoader
from .nmAPG import reconstruct_nmAPG
from .adam import reconstruct_adam
import torch
from deepinv.loss.metric import PSNR
from tqdm import tqdm

from torchvision.utils import save_image
import os
from PIL import Image
import time


def evaluate(
    physics,
    data_fidelity,
    dataset,
    regularizer,
    lmbd,
    NAG_step_size,
    NAG_max_iter,
    NAG_tol,
    adam=False,
    only_first=False,
    adaptive_range=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
    save_path=None,
    logger=None,
):

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    if logger is not None:
        logger.info(f"Number of test images: {len(dataloader)}.")
    if adaptive_range:
        psnr = PSNR(max_pixel=None)
    else:
        psnr = PSNR()

    regularizer.eval()
    for p in regularizer.parameters():
        p.requires_grad_(False)

    ## Evaluate on the test set
    psnrs = []
    iters = []
    Lip = []
    times = []  # List of recon durations for each test image
    x_out = None
    y_out = None
    recon_out = None
    for i, x in (progress_bar := tqdm(enumerate(dataloader))):
        x = x.to(torch.float32).to(device)
        y = physics(x)

        t_start = time.time()
        if adam:
            recon, stats = reconstruct_adam(
                y,
                physics,
                data_fidelity,
                regularizer,
                lmbd,
                NAG_step_size,
                NAG_max_iter,
                NAG_tol,
                verbose=verbose,
                return_stats=True,
            )
            stats["L"] = torch.tensor(0.0, dtype=torch.float, device=device)
        else:
            recon, stats = reconstruct_nmAPG(
                y,
                physics,
                data_fidelity,
                regularizer,
                lmbd,
                NAG_step_size,
                NAG_max_iter,
                NAG_tol,
                return_stats=True,
                verbose=False,
            )
        t_end = time.time()
        times.append(t_end - t_start)
        iters.append(stats["steps"])
        Lip.append(stats["L"].cpu())
        psnrs.append(psnr(recon, x).squeeze().item())

        if logger is not None:
            logger.info(f"Image {i} reconstructed, PSNR: {psnrs[-1]:.2f}")

        if save_path is not None and (i < 10):
            save_image(x, os.path.join(save_path, f"ground_truth_{i}.png"), padding=0)
            save_image(y, os.path.join(save_path, f"measurement_{i}.png"), padding=0)
            save_image(
                recon, os.path.join(save_path, f"reconstruction_{i}.png"), padding=0
            )

        if i == 0:
            y_out = y
            x_out = x
            recon_out = recon

        progress_bar.set_description(
            f"Mean PSNR: {np.mean(psnrs):.2f}, Last PSNR: {psnrs[-1]:.2f}, steps: {iters[-1]}"
        )
        if only_first:
            break
    mean_psnr = np.mean(psnrs)
    print_psnr = "Mean PSNR over the test set: {0:.2f}".format(mean_psnr)
    print(print_psnr)
    mean_iters = np.mean(iters)
    print_iters = "Mean iterations over the test set: {0:.2f}".format(mean_iters)
    print(print_iters)
    if not adam:
        mean_Lip = np.mean(Lip)
        print_Lip = "Mean L over the test set: {0:.2f}".format(mean_Lip)
        print(print_Lip)
    mean_time = np.mean(times)
    print_time = "Mean reconstruction time over the test set: {0:.2f} seconds".format(
        mean_time
    )
    print(print_time)

    if logger is not None:
        logger.info(print_psnr)
        logger.info(print_iters)
        if not adam:
            logger.info(print_Lip)
        logger.info(print_time)

    return mean_psnr, x_out, y_out, recon_out
