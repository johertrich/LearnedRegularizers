import numpy as np
from torch.utils.data import DataLoader
from .nmAPG import reconstruct_nmAPG
from .adam import reconstruct_adam
import torch
from deepinv.loss.metric import PSNR
from tqdm import tqdm

import matplotlib.pyplot as plt
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
    only_first=False,
    adaptive_range=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
    save_path=None,
    save_png=False,
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
    times = [] # List of recon durations for each test image
    x_out = None
    y_out = None
    recon_out = None
    for i, x in (progress_bar := tqdm(enumerate(dataloader))):

        if device == "mps":
            # mps does not support float64
            x = x.to(torch.float32).to(device)
        else:
            x = x.to(device).to(torch.float32)
        y = physics(x)
        
        t_start = time.time()
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
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 6))

            ax1.imshow(x[0, 0].cpu().numpy(), cmap="gray")
            ax1.axis("off")
            ax1.set_title("ground truth")

            ax2.imshow(recon[0, 0].cpu().numpy(), cmap="gray")
            ax2.axis("off")
            ax2.set_title("reconstruction")

            ax3.imshow(y[0, 0].cpu().numpy(), cmap="gray")
            ax3.axis("off")
            ax3.set_title("measurements")

            fig.suptitle(f"IDX={i} | PSNR {np.round(psnrs[-1],3)}")
            plt.savefig(os.path.join(save_path, f"imgs_{i}.png"))
            plt.close()

        if save_png and save_path is not None and (i < 10):

            # Scale to [0, 255] and convert to uint8
            image_uint8 = (recon[0, 0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

            # Create a PIL Image object
            image = Image.fromarray(image_uint8, mode="L")  # 'L' for grayscale

            # Save as a PNG file
            image.save(os.path.join(save_path, f"reco_{i}.png"))

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
    mean_Lip = np.mean(Lip)
    print_Lip = "Mean L over the test set: {0:.2f}".format(mean_Lip)
    print(print_Lip)
    mean_time = np.mean(times)
    print_time = "Mean reconstruction time over the test set: {0:.2f} seconds".format(mean_time)
    print(print_time)
    
    if logger is not None:
        logger.info(print_psnr)
        logger.info(print_iters)
        logger.info(print_Lip)
        logger.info(print_time)
    
    return mean_psnr, x_out, y_out, recon_out


def evaluate_adam(
    physics,
    data_fidelity,
    dataset,
    regularizer,
    lmbd,
    step_size,
    max_iter,
    tol,
    only_first=False,
    adaptive_range=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
    save_path=None,
    save_png=False,
):

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    if adaptive_range:
        psnr = PSNR(max_pixel=None)
    else:
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
            x = x.to(device).to(torch.float32)

        # def psnr_fun(rec):
        #    return psnr(rec, x).squeeze().item()

        y = physics(x)

        recon = reconstruct_adam(
            y,
            physics,
            data_fidelity,
            regularizer,
            lmbd,
            step_size,
            max_iter,
            tol,
            verbose=verbose,
            psnr_fun=None,
        )
        psnrs.append(psnr(recon, x).squeeze().item())

        if save_path is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 6))

            ax1.imshow(x[0, 0].cpu().numpy(), cmap="gray")
            ax1.axis("off")
            ax1.set_title("ground truth")

            ax2.imshow(recon[0, 0].cpu().numpy(), cmap="gray")
            ax2.axis("off")
            ax2.set_title("reconstruction")

            ax3.imshow(y[0, 0].cpu().numpy(), cmap="gray")
            ax3.axis("off")
            ax3.set_title("measurements")

            fig.suptitle(f"IDX={i} | PSNR {np.round(psnrs[-1],3)}")
            plt.savefig(os.path.join(save_path, f"imgs_{i}.png"))
            plt.close()

        if save_png and save_path is not None:

            # Scale to [0, 255] and convert to uint8
            image_uint8 = (recon[0, 0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)

            # Create a PIL Image object
            image = Image.fromarray(image_uint8, mode="L")  # 'L' for grayscale

            # Save as a PNG file
            image.save(os.path.join(save_path, f"reco_{i}.png"))

        if i == 0:
            y_out = y
            x_out = x
            recon_out = recon

        del y
        del recon
        del x

        if only_first:
            break
    mean_psnr = np.mean(psnrs)

    return mean_psnr, x_out, y_out, recon_out
