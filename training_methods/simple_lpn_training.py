"""
Created on Wed Mar 5 2025

@author: ZhenghanFang

Train LPN by proximal matching.

Source: https://github.com/ZhenghanFang/learned-proximal-networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

from priors.lpn.lpn import LPNPrior


def simple_lpn_training(
    regularizer: LPNPrior,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    *,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int = 40000,
    optimizer: str = "adam",
    sigma_noise: float = 0.1,
    validate_every_n_steps: int = 1000,
    writer=None,
    num_steps_pretrain: int = 20000,
    num_stages: int = 4,
    sigma_min: float = 0.08 * np.sqrt(128**2 * 3),
    pretrain_lr: float = 1e-3,
    lr: float = 1e-4,
):
    """
    Args:
        num_steps: Total number of training iterations
        sigma_noise: noise std in prox matching
        validate_every_n_steps: validate every n iterations
        writer: tensorboard writer
        num_steps_pretrain: Number of iterations for L1 loss pretraining
        num_stages: Number of stages in prox matching loss schedule
        sigma_min: Minimum value for gamma in prox matching loss
            Default is 0.08 * sqrt(data dimension)
            For 128x128 RGB images, data dimension is 128^2 * 3
        pretrain_lr: Learning rate for L1 loss pretraining
        lr: Learning rate for prox matching
    """
    model = regularizer.lpn

    # Initialize the optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters())

    validator = Validator(val_dataloader, writer, sigma_noise)

    global_step = 0
    progress_bar = tqdm(total=num_steps)
    progress_bar.set_description(f"Train")
    while True:
        for step, batch in enumerate(train_dataloader):
            if validate_every_n_steps > 0 and global_step % validate_every_n_steps == 0:
                validator.validate(model, global_step)

            model.train()
            # get loss hyperparameters and learning rate
            args = argparse.Namespace()
            args.num_steps = num_steps
            args.num_steps_pretrain = num_steps_pretrain
            args.num_stages = num_stages
            args.sigma_min = sigma_min
            args.pretrain_lr = pretrain_lr
            args.lr = lr
            loss_hparams, lr = get_loss_hparams_and_lr(args, global_step)

            # get loss
            loss_func = get_loss(loss_hparams)
            # set learning rate
            for g in optimizer.param_groups:
                g["lr"] = lr

            # Train step
            result = train_step(model, optimizer, batch, loss_func, sigma_noise, device)
            loss = result["loss"]

            logs = {
                "loss": loss.detach().item(),
                "set": loss_hparams,
                "lr": lr,
            }
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

            global_step += 1
            if global_step >= num_steps:
                break

        if global_step >= num_steps:
            break

    progress_bar.close()
    return regularizer


def train_step(model, optimizer, batch, loss_func, sigma_noise, device):
    clean_images = batch["image"].to(device)
    noise = torch.randn_like(clean_images)
    if type(sigma_noise) == list:
        # uniform random noise level
        sigma_noise = (
            torch.rand(1).to(noise.device) * (sigma_noise[1] - sigma_noise[0])
            + sigma_noise[0]
        )
    noisy_images = clean_images + sigma_noise * noise
    out = model(noisy_images)

    loss = loss_func(out, clean_images)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.wclip()  # clip weights to non-negative values to ensure convexity

    result = {"loss": loss}
    return result


class Validator:
    """Class for validation."""

    def __init__(self, dataloader, writer, sigma_noise):
        """
        Args:
            writer: tensorboard writer
        """
        self.dataloader = dataloader
        self.writer = writer
        assert type(sigma_noise) == float
        self.sigma_noise = sigma_noise

    def _validate(self, model):
        """Validate the model on the validation set."""

        model.eval()
        device = next(model.parameters()).device

        psnr_list = []
        ssim_list = []
        for batch in self.dataloader:
            clean_images = batch["image"].to(device)
            noise = torch.randn_like(clean_images)
            noisy_images = clean_images + noise * self.sigma_noise
            out = model(noisy_images)

            psnr_, ssim_ = self.compute_metrics(clean_images, out)
            psnr_list.extend(psnr_)
            ssim_list.extend(ssim_)

        print(f"PSNR: {np.mean(psnr_list)}")
        print(f"SSIM: {np.mean(ssim_list)}")
        self.psnr_list = psnr_list
        self.ssim_list = ssim_list

    def compute_metrics(self, gt, out):
        """gt, out: batch, channel, height, width. torch.Tensor."""
        gt = gt.cpu().detach().numpy().transpose(0, 2, 3, 1)
        out = out.cpu().detach().numpy().transpose(0, 2, 3, 1)

        psnr_ = [skimage_psnr(gt_, out_, data_range=1.0) for gt_, out_ in zip(gt, out)]
        ssim_ = [
            skimage_ssim(gt_, out_, channel_axis=2, data_range=1.0)
            for gt_, out_ in zip(gt, out)
        ]

        return psnr_, ssim_

    def _log(self, step):
        """Log the validation metrics."""

        if self.writer is None:
            return
        self.writer.add_scalar("val/psnr", np.mean(self.psnr_list), step)
        self.writer.add_scalar("val/ssim", np.mean(self.ssim_list), step)

    def validate(self, model, step):
        """Validate the model and log the metrics."""

        self._validate(model)
        self._log(step)


##########################
# Utils for loss function
##########################
def get_loss_hparams_and_lr(args, global_step):
    """Get loss hyperparameters and learning rate based on training schedule.
    Parameters:
        args (argparse.Namespace): Arguments from command line.
        global_step (int): Current training step.
    """
    if global_step < args.num_steps_pretrain:
        loss_hparams, lr = {"type": "l1"}, args.pretrain_lr
    else:
        num_steps = args.num_steps - args.num_steps_pretrain
        step = global_step - args.num_steps_pretrain

        def _get_loss_hparams_and_lr(num_steps, step):
            num_steps_per_stage = num_steps // args.num_stages
            stage = step // num_steps_per_stage
            if stage >= args.num_stages:
                stage = args.num_stages - 1
            loss_hparams = {
                "type": "prox_matching",  # proximal matching
                "sigma": args.sigma_min * (2 ** (args.num_stages - 1 - stage)),
            }
            lr = args.lr
            return loss_hparams, lr

        loss_hparams, lr = _get_loss_hparams_and_lr(num_steps, step)

    return loss_hparams, lr


def get_loss(loss_hparams):
    """Get loss function from hyperparameters.
    Parameters:
        loss_hparams (dict): Hyperparameters for loss function.
    """
    if loss_hparams["type"] == "l1":
        return nn.L1Loss()
    elif loss_hparams["type"] == "prox_matching":
        return ExpDiracSrgt(sigma=loss_hparams["sigma"])
    else:
        raise NotImplementedError


# surrogate L0 loss: -exp(-(x/sigma)^2) + 1
def exp_func(x, sigma):
    return -torch.exp(-((x / sigma) ** 2)) + 1


class ExpDiracSrgt(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, input, target):
        """
        input, target: batch, *
        """
        bsize = input.shape[0]
        dist = (input - target).pow(2).reshape(bsize, -1).sum(1).sqrt()
        return exp_func(dist, self.sigma).mean()
