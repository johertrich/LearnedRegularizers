"""
This file provides a unified evaluation routine of a learned regularizer on some dataset by solving the
variational problem and reporting the mean PSNR (and some other stats).

The optimiser is chosen via the ``method`` argument and dispatched through the unified
``reconstruct`` function:

    'nmapg', 'lbfgs_batched'   -- natively batched solvers (one call per batch)
    'adam'                     -- Adam with cosine-annealing schedule
    'l-bfgs', 'cg'             -- generic optimisers, looped per sample
"""

import numpy as np
from torch.utils.data import DataLoader
from .reconstruct import reconstruct
import torch
from deepinv.loss.metric import PSNR
from tqdm import tqdm
from torchvision.utils import save_image
import os
import time


def evaluate(
    physics,
    data_fidelity,
    dataset,
    regularizer,
    lmbd,
    step_size,
    max_iter,
    tol,
    method="nmapg",
    only_first=False,
    adaptive_range=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
    save_path=None,
    logger=None,
    **kwargs,
):
    """Evaluate a learned regularizer on a dataset and report mean PSNR.

    Parameters
    ----------
    physics : deepinv physics object
        Defines the forward operator and noise model.
    data_fidelity : deepinv data fidelity object
        Data fidelity term of the variational problem.
    dataset : torch Dataset
        Test dataset; images are loaded one at a time (batch size 1).
    regularizer : object
        Regularizer passed to ``reconstruct``.
    lmbd : float
        Regularisation weight.
    step_size : float
        Initial step size; see ``reconstruct`` for per-method semantics.
    max_iter : int
        Maximum solver iterations per image.
    tol : float
        Relative iterate-change stopping tolerance.
    method : str
        Solver to use: ``'nmapg'``, ``'adam'``, ``'lbfgs_batched'``,
        ``'l-bfgs'``, or ``'cg'``.
    only_first : bool
        If True, evaluate only the first image (useful for quick checks).
    adaptive_range : bool
        If True, PSNR range is chosen adaptively (common for CT data).
    device : str
        Torch device string.
    verbose : bool
        Print per-iteration solver progress.
    save_path : str, optional
        Directory in which to save ground-truth, measurement, and
        reconstruction images (first 10 images only).
    logger : logging.Logger, optional
        Python logger for writing per-image PSNR records.
    **kwargs
        Additional method-specific hyperparameters forwarded verbatim to
        ``reconstruct``.  See ``reconstruct`` docstring for the full list of
        accepted keys per method:

        nmapg
            ``L_init``, ``rho``, ``delta``, ``eta``
        lbfgs_batched
            ``history_size``, ``c1``, ``backtrack``, ``max_ls``, ``gtol``
        adam
            (none beyond ``step_size``)
        l-bfgs / cg
            ``history_size`` (l-bfgs only, default 10), ``lr``,
            ``line_search_variant`` (l-bfgs only, ``'strong'`` or ``'weak'``,
            default ``'strong'``), ``c1``, ``c2``, ``tolerance_change``,
            ``max_ls`` (all l-bfgs only), ``gtol`` (relative: ``‖g_k‖/‖g_0‖``),
            ``gtd_tol`` (l-bfgs only)

    Returns
    -------
    mean_psnr : float
    x_out : Tensor  -- first ground-truth image
    y_out : Tensor  -- first measurement
    recon_out : Tensor  -- first reconstruction
    """
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
    for i, x in (progress_bar := tqdm(enumerate(dataloader), total=len(dataloader))):
        x = x.to(torch.float32).to(device)
        y = physics(x)
        t_start = time.time()
        recon, stats = reconstruct(
            y,
            physics,
            data_fidelity,
            regularizer,
            lmbd,
            step_size,
            max_iter,
            tol,
            method=method,
            return_stats=True,
            verbose=verbose,
            **kwargs,
        )
        t_end = time.time()
        times.append(t_end - t_start)
        iters.append(stats["steps"])
        if (
            stats["L"] is not None
        ):  # some methods (e.g. Adam) report no Lipschitz estimate
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
    mean_iters = np.mean(iters)
    mean_time = np.mean(times)
    lines = [
        f"Mean PSNR over the test set: {mean_psnr:.2f}",
        f"Mean iterations over the test set: {mean_iters:.2f}",
        f"Mean reconstruction time over the test set: {mean_time:.2f} seconds",
        *((f"Mean L over the test set: {np.mean(Lip):.2f}") if Lip else ()),
    ]
    for line in lines:
        print(line)
        if logger is not None:
            logger.info(line)
    return mean_psnr, x_out, y_out, recon_out
