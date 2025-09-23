"""
Adversarial Regularization (AR) Training Module.

This module implements adversarial training for learned regularizers using a 
Wasserstein GAN with gradient penalty (WGAN-GP) framework. The regularizer 
is trained to distinguish between clean and noisy/corrupted images while 
maintaining appropriate regularization properties.

The training follows the adversarial regularization approach where:
- The regularizer acts as a discriminator between clean and corrupted data
- Gradient penalty ensures Lipschitz constraint satisfaction
- Validation is performed using iterative reconstruction algorithms

Based on: 
    https://arxiv.org/abs/1805.11572
    https://arxiv.org/abs/2008.02839


Functions:
    WGAN_loss: Computes WGAN-GP loss with gradient penalty
    estimate_lmbd: Estimates regularization parameter lambda
    estimate_lip: Estimates Lipschitz constant of the regularizer
    ar_training: Main training loop for adversarial regularization
"""

import torch
import copy
from tqdm import tqdm
import numpy as np
from evaluation import reconstruct_nmAPG
from deepinv.utils import patch_extractor
from deepinv.loss.metric import PSNR


def WGAN_loss(regularizer, images, images_gt, mu=10):
    """
    Compute the Wasserstein GAN loss with gradient penalty for adversarial regularization.
    
    The regularizer acts as a discriminator, and the loss consists of two parts:
    1. Wasserstein distance between distributions of clean and corrupted images
    2. Gradient penalty to enforce the Lipschitz constraint
    
    The gradient penalty ensures that the regularizer satisfies the 1-Lipschitz
    constraint, which is crucial for theoretical guarantees in variational problems.
    
    Args:
        regularizer: The regularizer network (acting as discriminator)
        images (torch.Tensor): Corrupted/noisy images (fake samples)
        images_gt (torch.Tensor): Clean ground truth images (real samples)
        mu (float, optional): Weight for gradient penalty term. Defaults to 10.
        
    Returns:
        tuple: (total_loss, gradient_penalty_loss)
            - total_loss (torch.Tensor): Combined Wasserstein + gradient penalty loss
            - gradient_penalty_loss (torch.Tensor): Gradient penalty component only
    """
    real_samples = images_gt
    fake_samples = images

    # Generate random interpolation points between real and fake samples
    B = real_samples.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=real_samples.device)
    interpolates = images_gt + alpha * (images - images_gt)
    interpolates.requires_grad_(True)
    
    # Compute gradient norm at interpolation points
    grad_norm = regularizer.grad(interpolates).flatten(1).norm(2, dim=1)
    
    # Wasserstein distance: E[D(real)] - E[D(fake)]
    data_loss = regularizer.g(real_samples).mean() - regularizer.g(fake_samples).mean()
    
    # Gradient penalty: penalize deviation from unit gradient norm
    grad_loss = mu * torch.nn.functional.relu(grad_norm - 1).square().mean()
    
    return data_loss + grad_loss, grad_loss


def estimate_lmbd(dataset, physics, device):
    """
    Estimate the regularization parameter lambda based on data consistency.
    
    This function estimates an appropriate value for the regularization parameter
    by computing the average residual norm of the data consistency term across
    the dataset. The regularization parameter balances data fidelity and
    regularization in variational formulations.
    
    The estimation is based on the principle that lambda should be scaled
    according to the typical magnitude of the data fidelity gradient, for a regularizer that is 1 Lipschitz.
    lambda ~ ||A^T(Ax - y)||_2
    
    Args:
        dataset: PyTorch dataset containing validation images, or None
        physics: Physics operator defining the forward model A
        device (str or torch.device): Device for computation
        
    Returns:
        float: Estimated regularization parameter lambda
        
    Note:
        If dataset is None, returns a default value of 1.0
    """
    if dataset is None:
        lmbd = 1.0
    else:
        with torch.no_grad():
            residual = 0.0
            for x in tqdm(dataset, desc="Estimating lambda"):
                x = x.to(device)
                y = physics(x)  # Forward model: Ax + noise
                # Compute data fidelity gradient: A^T(Ax - y)
                residual += torch.norm(
                    physics.A_adjoint(y - physics.A(x)), dim=(-2, -1)
                ).mean()
            lmbd = residual / (len(dataset))
        print("Estimated lambda: " + str(lmbd.item()))
    return lmbd


def estimate_lip(regularizer, dataset, device):
    """
    Estimate the Lipschitz constant of the regularizer.
    
    This function estimates the Lipschitz constant by computing the maximum
    gradient norm of the regularizer across the dataset. The Lipschitz constant
    is crucial for setting appropriate regularization strength, as lambda needs to be scaled
    in accordance with the Lipschitz constant of the regularizer.
    
    Args:
        regularizer: The regularizer network
        dataset: PyTorch dataset containing validation images, or None  
        device (str or torch.device): Device for computation
        
    Returns:
        torch.Tensor: Estimated Lipschitz constant (maximum gradient norm)
        
    Note:
        If dataset is None, returns a default value of 1.0.
        Also prints both maximum and average gradient norms for analysis.
    """
    if dataset is None:
        lip = 1.0
    else:
        with torch.no_grad():
            lip_avg = torch.tensor(0.0, device=device)
            lip_max = torch.tensor(0.0, device=device)
            for x in tqdm(dataset, desc="Estimating Lipschitz constant"):
                x = x.to(device)
                # Compute L2 norm of gradient
                gradients = torch.sqrt(torch.sum(regularizer.grad(x) ** 2))
                lip_avg += gradients
                lip_max = torch.max(lip_max, gradients)
            lip_avg = lip_avg / len(dataset)
        print(
            "Lipschitz constant: Max "
            + str(lip_max.item())
            + " Avg "
            + str(lip_avg.item())
        )
    return lip_max


def ar_training(
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
    LAR_eval=False,
    dynamic_range_psnr=False,
    savestr=None,
    logger=None,
):
    """
    Train a regularizer using adversarial regularization (AR) with WGAN-GP loss.
    
    This function implements the adversarial regularization training procedure where
    a regularizer network is trained to distinguish between clean and corrupted images
    while satisfying the Lipschitz constraint through gradient penalty. The training
    alternates between adversarial loss minimization and validation using iterative
    reconstruction.
    
    The key components of AR training:
    1. Adversarial loss with gradient penalty (WGAN-GP)
    2. Patch-based training for computational efficiency
    3. Validation using iterative reconstruction algorithms
    4. Lipschitz constant estimation for proper step size scaling
    
    Args:
        regularizer: The regularizer network to be trained
        physics: Physics operator defining the forward model (A, A_dagger, etc.)
        data_fidelity: Data fidelity term for reconstruction
        train_dataloader (DataLoader): Training data loader
        val_dataloader (DataLoader): Validation data loader  
        lmbd (float, optional): Regularization parameter. If None, estimated automatically.
        epochs (int, optional): Total training epochs. Defaults to 1000.
        validation_epochs (int, optional): Frequency of validation. Defaults to 100.
        lr (float, optional): Initial learning rate. Defaults to 1e-3.
        lr_decay (float, optional): Exponential learning rate decay factor. Defaults to 0.998.
        device (str, optional): Computing device. Defaults to auto-detected GPU/CPU.
        mu (float, optional): Gradient penalty weight in WGAN-GP loss. Defaults to 10.0.
        patch_size (int, optional): Size of patches for training. If None, uses full images.
        patches_per_img (int, optional): Number of patches per image. Defaults to 8.
        LAR_eval (bool, optional): Whether using Local AR evaluation mode. Defaults to False.
        dynamic_range_psnr (bool, optional): Use dynamic range for PSNR calculation. Defaults to False.
        savestr (str, optional): Path prefix for saving checkpoints. If None, no saving.
        logger (logging.Logger, optional): Logger for training progress. If None, uses print.
        
    Returns:
        regularizer: The trained regularizer with best validation performance loaded
        
    Raises:
        AssertionError: If validation_epochs > epochs
        
    Note:
        The function saves the best model based on validation PSNR and loads it
        before returning. Training progress is logged including loss values,
        PSNR scores, and Lipschitz constant estimates.
    """
    # Validate input parameters
    assert validation_epochs <= epochs, (
        "validation_epochs cannot be greater than epochs. "
        "If validation_epochs > epochs, no validation will occur, "
        "best_regularizer_state will remain unchanged, and the returned model will be identical to the initial state."
    )

    # Setup PSNR metric for validation
    if dynamic_range_psnr:
        psnr = PSNR(max_pixel=None)  # Dynamic range based on image content
    else:
        psnr = PSNR()  # Fixed range [0, 1]

    # Estimate regularization parameter if not provided
    if lmbd is None:
        lmbd = estimate_lmbd(val_dataloader, physics, device)

    # Setup reconstruction algorithm parameters
    NAG_step_size = 1e-2  # Step size for Nesterov accelerated gradient
    NAG_max_iter = 1000   # Maximum iterations for reconstruction
    NAG_tol_val = 1e-4    # Tolerance for convergence

    # Setup training components
    adversarial_loss = WGAN_loss
    optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    best_val_psnr = -999  # Track best validation performance

    # ============================================================================
    # Main Training Loop
    # ============================================================================
    for epoch in range(epochs):
        loss_vals = []
        grad_loss_vals = []
        
        # Training phase: Iterate over training batches
        for x in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            x = x.to(device)
            
            # Generate corrupted data using forward physics model
            y = physics(x)                    # y = Ax + noise (measurement)
            x_noisy = physics.A_dagger(y)     # x_noisy = A†y (corrupted reconstruction)
            
            # Choose between full-image or patch-based training
            if patch_size is None or (x.shape[-1] == patch_size and x.shape[-2] == patch_size):
                # Full-image training
                loss, grad_loss = adversarial_loss(regularizer, x_noisy, x, mu)
            else:
                # Patch-based training for computational/learning efficiency
                # Extract patches from ground truth images
                x_patches, linear_inds = patch_extractor(
                    x, n_patches=patches_per_img, patch_size=patch_size
                )
                
                # Extract corresponding patches from corrupted images
                B, C, _, _ = x_noisy.shape
                imgs = x_noisy.reshape(B, -1)
                x_noisy_patches = imgs.view(B, -1)[:, linear_inds]
                x_noisy_patches = x_noisy_patches.reshape(
                    patches_per_img * x.shape[0], C, patch_size, patch_size
                )
                x_patches = x_patches.reshape(
                    patches_per_img * x.shape[0], C, patch_size, patch_size
                )
                
                # Apply adversarial loss on patches
                if LAR_eval:
                    # Local AR mode: use CNN component directly
                    loss, grad_loss = adversarial_loss(
                        regularizer.cnn, x_noisy_patches, x_patches, mu
                    )
                else:
                    # Standard mode: use full regularizer
                    loss, grad_loss = adversarial_loss(
                        regularizer, x_noisy_patches, x_patches, mu
                    )
            
            # Backpropagation and parameter update
            loss.backward()
            optimizer.step()
            
            # Track loss values for monitoring
            loss_vals.append(loss.item())
            grad_loss_vals.append(grad_loss.item())

        # Update learning rate
        scheduler.step()

        # Log training progress
        print_str = f"Average training loss in epoch {epoch + 1}: {np.mean(loss_vals):.2E}, average grad loss: {np.mean(grad_loss_vals):.2E}"
        print(print_str)
        if logger is not None:
            logger.info(print_str)
            
        # ========================================================================
        # Validation Phase (periodic)
        # ========================================================================
        if (epoch + 1) % validation_epochs == 0:
            regularizer.eval()  # Set to evaluation mode
            
            # Estimate Lipschitz constant for proper step size scaling
            if LAR_eval:
                # For Local AR: temporarily disable padding for Lipschitz estimation
                pad = regularizer.pad
                regularizer.pad = False
                lip = estimate_lip(regularizer, val_dataloader, device)
                regularizer.pad = pad
            else:
                lip = estimate_lip(regularizer, val_dataloader, device)
            
            # Validation using iterative reconstruction
            with torch.no_grad():
                if LAR_eval:
                    # Enable padding for reconstruction
                    pad = regularizer.pad
                    regularizer.pad = True
                    
                val_psnr_epoch = 0  # Track validation PSNR
                
                # Iterate over validation dataset
                for x_val in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    x_val = x_val.to(device).to(torch.float32)
                    y_val = physics(x_val)              # Generate measurement
                    x_val_noisy = physics.A_dagger(y_val)  # Initial estimate
                    
                    # Perform iterative reconstruction with learned regularizer
                    x_recon_val = reconstruct_nmAPG(
                        y_val,                # Measurements
                        physics,              # Forward model
                        data_fidelity,        # Data fidelity term
                        regularizer,          # Learned regularizer
                        lmbd / lip,          # Scaled regularization parameter
                        NAG_step_size,       # Step size for optimization
                        NAG_max_iter,        # Maximum iterations
                        NAG_tol_val,         # Convergence tolerance
                        verbose=False,
                        x_init=x_val_noisy,  # Initial estimate
                    )
                    
                    # Compute PSNR for validation
                    new_psnr = psnr(x_recon_val, x_val).mean().item()
                    if new_psnr <= 0:
                        print(f"Warning: Negative PSNR occurred: {new_psnr}")
                    val_psnr_epoch += new_psnr

                # Restore LAR padding setting if needed
                if LAR_eval:
                    regularizer.pad = pad
                    
                # Compute average validation PSNR
                mean_val_psnr = val_psnr_epoch / len(val_dataloader)
                print_str = f"[Epoch {epoch+1}] Validation PSNR: {mean_val_psnr:.2f}"
                print(print_str)

                # Save checkpoint if savestr is provided
                if savestr is not None:
                    torch.save(
                        regularizer.state_dict(),
                        savestr + "_epoch_" + str(epoch) + ".pt",
                    )

                if logger is not None:
                    logger.info(print_str)

                # Update best model based on validation PSNR
                if mean_val_psnr > best_val_psnr:
                    print("Updated best PSNR")
                    best_val_psnr = mean_val_psnr
                    best_regularizer_state = copy.deepcopy(regularizer.state_dict())
    
    # ============================================================================
    # Training Complete: Load Best Model
    # ============================================================================
    print_str = f"Training completed. Best validation PSNR: {best_val_psnr:.2f}"
    print(print_str)
    if logger is not None:
        logger.info(print_str)
    
    # Load the best performing model before returning
    regularizer.load_state_dict(best_regularizer_state)
    return regularizer
