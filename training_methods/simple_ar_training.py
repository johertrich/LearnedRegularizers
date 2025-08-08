import torch
import copy
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from evaluation import reconstruct_nmAPG
from dataset import get_dataset
from torchvision.transforms import CenterCrop, RandomCrop
import torchvision.transforms.functional as TF
from deepinv.utils import patch_extractor
from torch.utils.data import DataLoader
from deepinv.loss.metric import PSNR


def WGAN_loss(regularizer, images, images_gt,mu=10):
    """Calculates the gradient penalty loss for WGAN GP"""
    real_samples=images_gt
    fake_samples=images

    B = real_samples.size(0) 
    alpha = torch.rand(B, 1, 1, 1, device=real_samples.device)
    interpolates = images_gt + alpha * (images - images_gt)
    interpolates.requires_grad_(True)
    grad_norm = regularizer.grad(interpolates).flatten(1).norm(2, dim=1) 
    data_loss = regularizer.g(real_samples).mean() - regularizer.g(fake_samples).mean()
    grad_loss = mu * torch.nn.functional.relu(grad_norm - 1).square().mean()
    return data_loss + grad_loss,  grad_loss

def estimate_lmbd(dataset,physics,device):
    if dataset is None: lmbd=1.0
    else: 
        with torch.no_grad():
            residual = 0.0
            for x in tqdm(dataset, total=len(dataset)):
                x = x.to(device)
                y = physics(x) ##Ax+e
                residual += torch.norm(physics.A_adjoint(y - physics.A(x)),dim=(-2,-1)).mean()
            lmbd = residual/(len(dataset))
        print('Estimated lambda: ' + str(lmbd.item()))
    return lmbd

def estimate_lip (regularizer,dataset,device):
    if dataset is None: lip=1.0
    else:
        with torch.no_grad():
            lip_avg = torch.tensor(0.0, device=device)
            lip_max = torch.tensor(0.0, device=device)
            for x in tqdm(dataset, total=len(dataset)):
                x = x.to(device)
                gradients = torch.sqrt(torch.sum(regularizer.grad(x)**2))
                lip_avg += gradients
                lip_max = torch.max(lip_max, gradients)
            lip_avg = lip_avg/len(dataset)
        print('Lipschitz constant: Max ' + str(lip_max.item()) +  ' Avg ' + str(lip_avg.item()))
    return lip_max

def simple_ar_training(
    regularizer,
    physics,
    data_fidelity,
    train_dataloader,
    val_dataloader,
    lmbd=None,
    epochs=1000,
    validation_epochs=100,
    lr=1e-3,
    lr_decay=0.998,
    device="cuda" if torch.cuda.is_available() else "cpu",
    mu=10.0,
    patch_size=None,
    patches_per_img=8,
    LAR_eval = False,
    dynamic_range_psnr=False,
    savestr=None,
    logger=None,
):
    assert validation_epochs <= epochs, (
        "validation_epochs cannot be greater than epochs. "
        "If validation_epochs > epochs, no validation will occur, "
        "best_regularizer_state will remain unchanged, and the returned model will be identical to the initial state."
    )
    
    if dynamic_range_psnr:
        psnr = PSNR(max_pixel=None)
    else:
        psnr = PSNR()

    if lmbd == None:
        lmbd = estimate_lmbd(val_dataloader,physics,device)

    NAG_step_size=1e-2#/lmbd
    NAG_max_iter=1000
    NAG_tol_val=1e-4

    adversarial_loss = WGAN_loss
    optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    best_val_psnr=-999

    for epoch in range(epochs):
        loss_vals = []
        grad_loss_vals = []
        for x in tqdm(train_dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            y = physics(x)
            x_noisy = physics.A_dagger(y)
            if not patch_size == None:
                x_patches, linear_inds = patch_extractor(x, n_patches=patches_per_img, patch_size=patch_size)
                B, C, _, _ = x_noisy.shape
                imgs = x_noisy.reshape(B, -1)
                x_noisy_patches = imgs.view(B, -1)[:, linear_inds]
                x_noisy_patches = x_noisy_patches.reshape(patches_per_img*x.shape[0], C, patch_size, patch_size)
                x_patches = x_patches.reshape(patches_per_img*x.shape[0], C, patch_size, patch_size)
                if LAR_eval:
                    loss, grad_loss = adversarial_loss(regularizer.cnn, x_noisy_patches, x_patches, mu)
                else:
                    loss, grad_loss = adversarial_loss(regularizer, x_noisy_patches, x_patches, mu)
            else:
                loss, grad_loss = adversarial_loss(regularizer, x_noisy, x, mu)
            loss.backward()
            optimizer.step()
            loss_vals.append(loss.item())
            grad_loss_vals.append(grad_loss.item())

        scheduler.step()

        print_str = f"Average training loss in epoch {epoch + 1}: {np.mean(loss_vals):.2E}, average grad loss: {np.mean(grad_loss_vals):.2E}"
        print(print_str)
        if logger is not None:
            logger.info(print_str)
        if (epoch + 1) % validation_epochs == 0:
            regularizer.eval()
            lip = estimate_lip(regularizer,val_dataloader,device)
            with torch.no_grad():
                val_loss_epoch = 0
                val_psnr_epoch = 0
                for x_val in tqdm(
                    val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Val"
                ):
                    x_val = x_val.to(device).to(torch.float32)
                    y_val = physics(x_val)
                    x_val_noisy = physics.A_dagger(y_val)

                    x_recon_val = reconstruct_nmAPG(
                        y_val,
                        physics,
                        data_fidelity,
                        regularizer,
                        lmbd/lip,
                        NAG_step_size,
                        NAG_max_iter,
                        NAG_tol_val,
                        verbose=False,
                        x_init=x_val_noisy,
                    )

                    #import matplotlib.pyplot as plt 
                    #fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
                    #ax1.imshow(x_val[0,0].cpu().numpy(), cmap="gray")
                    #ax2.imshow(x_recon_val[0,0].cpu().numpy(), cmap="gray")
                    #plt.show()

                    new_psnr = psnr(x_recon_val, x_val).mean().item()
                    if new_psnr <= 0:
                        print(f"Warning: Negativ PSNR occured {new_psnr}")
                    val_psnr_epoch += new_psnr

                mean_val_psnr = val_psnr_epoch / len(val_dataloader)
                print_str = f"[Epoch {epoch+1}] PSNR: {mean_val_psnr:.2f}"
                print(print_str)

                if savestr is not None:
                    torch.save(
                        regularizer.state_dict(),
                        savestr + "_epoch_" + str(epoch) + ".pt",
                    )

                if logger is not None:
                    logger.info(print_str)

                # ---- Save best regularizer based on validation PSNR ----
                if mean_val_psnr > best_val_psnr:
                    print("Updated best PSNR")
                    best_val_psnr = mean_val_psnr
                    best_regularizer_state = copy.deepcopy(regularizer.state_dict())
    # Load best regularizer
    print_str = f"Best Training PSNR: {best_val_psnr}"
    print(print_str)
    if logger is not None:
        logger.info(print_str)
    regularizer.load_state_dict(best_regularizer_state)
    return regularizer
