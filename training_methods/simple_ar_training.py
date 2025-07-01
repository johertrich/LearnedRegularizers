import torch
import copy
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from evaluation import reconstruct_nmAPG
from dataset import get_dataset
from torchvision.transforms import CenterCrop, RandomCrop
from deepinv.utils import patch_extractor
from torch.utils.data import DataLoader
from deepinv.loss.metric import PSNR


def WGAN_loss(regularizer, images, images_gt,mu=10):
    """Calculates the gradient penalty loss for WGAN GP"""
    real_samples=images_gt
    fake_samples=images
    
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).type_as(real_samples)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    grad_norm = torch.linalg.vector_norm(regularizer.grad(interpolates), dim=(1,2,3))
    data_loss = regularizer.g(real_samples).mean() - regularizer.g(fake_samples).mean()
    grad_loss = mu*(torch.clip(grad_norm - 1, min=0.) ** 2).mean()
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
    mu = 10.0,
    dynamic_range_psnr=False,
    savestr=None,
    logger=None,
):
    assert validation_epochs <= epochs, (
        "validation_epochs cannot be greater than epochs. "
        "If validation_epochs > epochs, no validation will occur, "
        "best_regularizer_state will remain unchanged, and the returned model will be identical to the initial state."
    )
    
    NAG_step_size=1e-1
    NAG_max_iter=1000
    NAG_tol_val=1e-4

    if dynamic_range_psnr:
        psnr = PSNR(max_pixel=None)
    else:
        psnr = PSNR()

    if lmbd == None:
        lmbd = estimate_lmbd(val_dataloader,physics,device)
    
    adversarial_loss = WGAN_loss
    optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    
    for epoch in range(epochs):
        loss_vals = []
        grad_loss_vals = []
        for x in tqdm(train_dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            y = physics(x)
            x_noisy = physics.A_dagger(y)
            
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
        best_val_psnr=-999
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

                    val_psnr_epoch += psnr(x_recon_val, x_val).mean().item()

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
                    best_val_psnr = mean_val_psnr
                    best_regularizer_state = copy.deepcopy(regularizer.state_dict())
    # Load best regularizer
    regularizer.load_state_dict(best_regularizer_state)

    return regularizer

# Training function for the LocalAR
def simple_lar_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_data,
    val_data,
    patch_size,
    epochs=25,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    mu = 10.0,
    batch_size=128,
    save_str=None,
    val_epochs = 5,
    dataset_name="BSD500"
):
    adversarial_loss = WGAN_loss
    regularizer.to(device)
    optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100. )
    
    regularizer.train()
    NAG_step_size = 1e-2  # step size in NAG
    NAG_max_iter = 200  # maximum number of iterations in NAG
    NAG_tol = 1e-4  # tolerance for therelative error (stopping criterion)
    only_first = False
    def eval_routine():
        mean_psnr, x_out, y_out, recon_out = evaluate(
                    physics=physics,
                    data_fidelity=data_fidelity,
                    dataset=val_data, 
                    regularizer=regularizer,
                    lmbd=lmbd,
                    NAG_step_size=NAG_step_size,
                    NAG_max_iter=NAG_max_iter,
                    NAG_tol=NAG_tol,
                    only_first=only_first,
                    device=device,
                    verbose=False,
                    adaptive_range=True if dataset_name == "LoDoPab" else False 
                )
        for p in regularizer.parameters():
            p.requires_grad_(True)
        return mean_psnr, x_out, y_out, recon_out
    mean_psnr, x_out, y_out, recon_out = eval_routine()
    
    print("PSNR of initial model: ", mean_psnr)
    
    best_psnr = mean_psnr
    for epoch in tqdm(range(epochs)):
        loss_vals = []
        regularizer.train()

        train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
        for x in train_dataloader:
            optimizer.zero_grad()
            if isinstance(x, list):
                x = x[0]
            if device == "mps":
                x = x.to(torch.float32).to(device)
            else:
                x = x.to(device).to(torch.float)
            y = physics(x)
            x_noisy = physics.A_dagger(y)

            ### patch-based training
            x_patches, linear_inds = patch_extractor(x, n_patches=batch_size, patch_size=patch_size)
            x_patches = x_patches.squeeze(0)

            _, C, _, _ = x_noisy.shape
            imgs = x_noisy.reshape(1, -1)
            x_noisy_patches = imgs.view(1, -1)[:, linear_inds]
            x_noisy_patches = x_noisy_patches.reshape(batch_size, C, patch_size, patch_size)

            loss = adversarial_loss(regularizer.cnn, x_noisy_patches, x_patches, mu)

            mean_loss = torch.mean(loss)
            mean_loss.backward()
            optimizer.step()
            loss_vals.append(mean_loss.item())
        print(
            "Average training loss in epoch {0}: {1:.2E}".format(
                epoch + 1, np.mean(loss_vals)
            )
        )
        
        loss_vals = []
        
        
        scheduler.step()    
        
        print("Learning rate: ", scheduler.get_last_lr()[0])
        if epoch % val_epochs == 0 and epoch > 0:
            mean_psnr, x_out, y_out, recon_out = eval_routine()

            print("Mean val PSNR: ", mean_psnr)
            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                print("New best PSNR: ", best_psnr)
                if save_str is not None: 
                    torch.save(regularizer.cnn.state_dict(), save_str)
