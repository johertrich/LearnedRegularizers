import torch
import numpy as np
from tqdm import tqdm
from deepinv.physics import Denoising
from deepinv.loss.metric import PSNR
from torch.utils.data import RandomSampler, Dataset, Subset
from deepinv.optim.utils import minres
from evaluation import reconstruct_nmAPG
from PIL import Image
from collections import deque
import torch.nn as nn
import torchvision.transforms as transforms
import copy

# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# np.random.seed(0)
# random.seed(0)


class GaussianNoise_MAID(torch.nn.Module):
    def __init__(self, sigma=0.1, rng: torch.Generator = torch.default_generator):
        super().__init__()
        self.sigma = sigma
        self.rng = rng
        self.noise = None
        self.noise_validation = None

    def forward(self, x):
        """
        Adds Gaussian noise to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Noisy tensor.
        """
        if self.noise == None:
            self.noise = torch.randn_like(x) * self.sigma
            self.noise = self.noise.cpu().detach()
            noise = self.noise.to(x.device)
        elif self.noise.shape != x.shape:
            if self.noise_validation == None:
                self.noise_validation = torch.randn_like(x) * self.sigma
                self.noise_validation = self.noise_validation.cpu().detach()
            noise = self.noise_validation.to(x.device)
        else:
            noise = self.noise.to(x.device)
        return x + noise


# --- Helper Function to Extract Patches Deterministically ---
def extract_patches(image_tensor, patch_size, stride):
    """
    Extracts patches from a single image tensor deterministically.

    Args:
        image_tensor (torch.Tensor): Input image tensor (C, H, W).
        patch_size (int): The height and width of the patches.
        stride (int): The step size between patches.

    Returns:
        list[torch.Tensor]: A list of patch tensors.
    """
    patches = []
    _, h, w = image_tensor.shape
    # Iterate over the image grid with the specified stride
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image_tensor[:, i : i + patch_size, j : j + patch_size]
            patches.append(patch)
    return patches


# --- Custom Dataset for Patches (Patching Augmentation for MAID) ---
class PatchesDataset(Dataset):
    """
    A dataset that holds deterministic patches extracted from an original dataset.
    """

    def __init__(self, original_dataset, patch_size, stride, transform=None):
        """
        Args:
            original_dataset (Dataset): The original dataset (e.g., BSDS500).
            patch_size (int): The size of the patches to extract.
            stride (int): The stride for patch extraction.
            transform (callable, optional): Optional transform to be applied
                                             *after* patch extraction.
        """
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.all_patches = []
        self.original_image_indices = []  # Optional: track origin

        print(f"Processing original dataset to extract patches...")
        pre_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
                transforms.ToTensor(),  # Convert PIL Image to [C, H, W] tensor [0,1]
            ]
        )

        num_original_images = len(original_dataset)
        for idx in range(num_original_images):
            original_image = original_dataset[idx]  # Get image, ignore label if any

            # Ensure it's a PIL image before applying pre_transform if needed
            if isinstance(original_image, Image.Image):
                image_tensor = pre_transform(original_image)
            else:
                image_tensor = original_image
            # Check if image is large enough for at least one patch
            c, h, w = image_tensor.shape
            if h < patch_size or w < patch_size:
                print(
                    f"Warning: Image {idx} (size {h}x{w}) is smaller than patch size {patch_size}x{patch_size}. Skipping."
                )
                continue

            # Extract patches for the current image
            patches = extract_patches(image_tensor, self.patch_size, self.stride)
            self.all_patches.extend(patches)

            # Optional: Store which original image each patch came from
            self.original_image_indices.extend([idx] * len(patches))

            if (idx + 1) % 50 == 0 or (idx + 1) == num_original_images:
                print(f"  Processed {idx + 1}/{num_original_images} original images...")

        print(f"Finished extracting patches. Total patches: {len(self.all_patches)}")

    def __len__(self):
        """Returns the total number of patches."""
        return len(self.all_patches)

    def __getitem__(self, idx):
        """Returns the patch at the given index."""
        patch = self.all_patches[idx]
        # Optional: get original image index: original_idx = self.original_image_indices[idx]

        if self.transform:
            patch = self.transform(patch)  # Apply any post-patching transforms

        # Return patch (and optionally original_idx if needed later)
        return patch  # , original_idx


class GradientDescent(nn.Module):
    """
    Simple gradient descent module for upper-level problem in the bilevel optimization.
    This module applies a gradient descent step to the parameters of the regularizer.
    """

    def __init__(self, regularizer, lr=1e-3):
        super(GradientDescent, self).__init__()
        self.regularizer = regularizer
        self.lr = lr

    def zero_grad(self):
        """
        Zero out the gradients of the regularizer parameters.
        """
        for param in self.regularizer.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def forward(self):
        """
        Apply gradient descent step to the regularizer parameters.
        """
        with torch.no_grad():
            for param in self.regularizer.parameters():
                if param.grad is not None:
                    param -= self.lr * param.grad


class AdaGrad(nn.Module):
    """
    Manual AdaGrad optimizer module for upper-level problem in the bilevel optimization.
    Applies AdaGrad update to the regularizer's parameters.
    """

    def __init__(self, regularizer, lr=1e-2, eps=1e-8, window_size=0):
        super(AdaGrad, self).__init__()
        self.regularizer = regularizer
        self.lr = lr
        self.eps = eps
        self.window_size = (
            window_size  # Optional: for truncated AdaGrad. We do not use it by default
        )
        # Initialize state for each parameter (accumulator for squared gradients)
        self._grad_squared_accum = {}
        self._momentum_buffer = {}  # Optional momentum buffer
        for name, param in self.regularizer.named_parameters():
            if param.requires_grad:
                if self.window_size > 0:
                    self._grad_squared_accum[name] = deque(maxlen=self.window_size)
                else:
                    self._grad_squared_accum[name] = torch.zeros_like(param.data)

    def zero_grad(self):
        """
        Zero out the gradients of the regularizer parameters.
        """
        for param in self.regularizer.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def forward(self):
        with torch.no_grad():
            for name, param in self.regularizer.named_parameters():
                if param.grad is not None:
                    grad = param.grad
                    if self.window_size > 0:
                        # Truncated AdaGrad: push to window
                        self._grad_squared_accum[name].append(grad.pow(2))
                        avg_sq_grad = torch.stack(
                            list(self._grad_squared_accum[name])
                        ).mean(dim=0)
                        adjusted_lr = self.lr / (avg_sq_grad.sqrt() + self.eps)
                    else:
                        # Standard AdaGrad
                        self._grad_squared_accum[name].add_(grad.pow(2))
                        adjusted_lr = self.lr / (
                            self._grad_squared_accum[name].sqrt() + self.eps
                        )

                    param -= adjusted_lr * grad


def preprocess(x, device):
    dtype = torch.float32 if device == "mps" else torch.float
    return x.to(dtype).to(device)


def bilevel_training_maid(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataset,
    PATCH_SIZE,
    STRIDE,
    SUBSET,
    val_dataloader,
    epochs=100,
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_train=1e-1,
    NAG_tol_val=1e-4,
    cg_max_iter=1000,
    CG_tol=1e-6,
    lr=1e-3,
    lr_decay=0.5,
    stopping_criterion=1e-10,
    reg=False,
    reg_para=1e-5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    precondition=False,
    verbose=False,
    validation_epochs=20,
    logger=None,
    optimizer=None,
):
    """
    Bilevel learning using inexact hypegradient methods described in
    https://academic.oup.com/imamat/article/89/1/254/7456287
    https://arxiv.org/abs/2308.10098
    https://arxiv.org/abs/2412.12049
    """
    # Setting the Noise Model of Physics to be deterministic (not redrawn at each call)
    noise_level = physics.noise_model.sigma
    physics_train = Denoising(noise_model=GaussianNoise_MAID(sigma=noise_level))
    
    def hessian_vector_product(
        x,
        v,
        data_fidelity,
        y,
        regularizer,
        lmbd,
        physics,
        diff=False,
        only_reg=False,
    ):
        x = x.requires_grad_(True)
        if only_reg:
            grad = lmbd * regularizer.grad(x)
        else:
            grad = data_fidelity.grad(x, y, physics) + lmbd * regularizer.grad(x)
        dot = torch.dot(grad.view(-1), v.view(-1))
        hvp = torch.autograd.grad(dot, x, create_graph=diff)[0]
        if diff:
            return hvp
        return hvp.detach()

    def jac_vector_product(x, v, data_fidelity, y, regularizer, lmbd, physics):
        grad_lower_level = lambda x: data_fidelity.grad(
            x, y, physics
        ) + lmbd * regularizer.grad(x)
        for param in regularizer.parameters():
            if param.requires_grad:
                dot = torch.dot(grad_lower_level(x).view(-1), v.view(-1))
                if param.grad is None:
                    param.grad = -torch.autograd.grad(dot, param, create_graph=False)[
                        0
                    ].detach()
                else:
                    param.grad -= torch.autograd.grad(dot, param, create_graph=False)[
                        0
                    ].detach()
        return regularizer
    
    def jac_pow_loss(x, M=50, tol=1e-2):
        hvp = torch.randint(low=0, high=1, size=x.shape).to(x) * 2 - 1
        hvp_old = hvp.clone()
        for i in range(M):
            hvp = hessian_vector_product(
                x,
                hvp,
                data_fidelity,
                y,
                regularizer,
                lmbd,
                physics_train,
                diff=False,
                only_reg=True,
            ).detach()
            hvp = torch.nn.functional.normalize(hvp, dim=[-2, -1], out=hvp)
            if torch.norm(hvp - hvp_old) / x.size(0) < tol:
                break
            hvp_old = hvp.clone()
        hvp = hvp.clone(memory_format=torch.contiguous_format).detach()
        hvp = hessian_vector_product(
            x,
            hvp,
            data_fidelity,
            y,
            regularizer,
            lmbd,
            physics_train,
            diff=True,
            only_reg=True,
        )
        norm_sq = torch.sum(hvp ** 2) / x.size(0)
        print(f"Jac_Loss: {norm_sq}")
        if logger is not None:
            logger.info(
                f"Jac Loss {norm_sq}"
            )
        return torch.clip(norm_sq, min=200, max=None)

    # Initialize optimizer for the upper-level
    if optimizer is None:
        if precondition:
            optimizer = AdaGrad(
                regularizer,
                lr=lr,
            )
        else:
            optimizer = GradientDescent(
                regularizer,
                lr=lr,
            )
    success = False  # Flag for backtracking line search success
    psnr = PSNR()  # PSNR metric definition
    patch_dataset = PatchesDataset(
        original_dataset=train_dataset,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        transform=None,  # Add post-patch transforms here if needed
    )
    num_patches_total = len(patch_dataset)
    num_subset_patches = SUBSET
    # Use a fixed range for deterministic subset selection
    subset_indices = torch.randint(0, num_patches_total, (num_subset_patches,))
    train_subset = Subset(patch_dataset, subset_indices)
    train_dataloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=num_subset_patches,
        shuffle=False,
        drop_last=True,
        num_workers=8,
    )
    
    #physics.noise_model = GaussianNoise_MAID(sigma=noise_level)
    # logging lists and dictionaries
    loss_vals = []
    psnr_vals = []
    loss_train = []
    loss_val = []
    psnr_train = []
    psnr_val = []
    
    best_val_psnr = -float("inf")
    best_regularizer_state = copy.deepcopy(regularizer.state_dict())
    
    # Hyperparameters for the MAID optimizer
    rho_maid = lr_decay
    nu_over = 1.05
    nu_under = 0.5
    rho_over = 1.25
    eps = NAG_tol_train
    eps_old = NAG_tol_train
    max_line_search = 5
    fixed_eps = False
    fixed_lr = False

    # Training loop
    for epoch in range(epochs):
        regularizer.train()

        # Check if the data loader is deterministic
        if len(train_dataloader) > 1:
            raise ValueError(
                "MAID optimizer is only compatible with full-batch training. "
                "Ensure the data loader is set up for a single batch containing the entire dataset."
            )

        if isinstance(train_dataloader.sampler, RandomSampler):
            raise ValueError(
                "MAID optimizer is only compatible with deterministic data loaders. "
                "Set shuffle=False in the DataLoader."
            )
        for x in tqdm(train_dataloader):
            x = preprocess(x, device)
            y = physics_train(x)
            if epoch == 0:
                x_init = y.detach().clone()
            else:
                # Warm start with the previous reconstruction for MAID
                if success:
                    x_init = x_recon.detach().clone()
            # Solve the lower-level problem to compute the hypergradient
            x_recon, stats = reconstruct_nmAPG(
                y,
                physics_train,
                data_fidelity,
                optimizer.regularizer,
                lmbd,
                NAG_step_size,
                NAG_max_iter,
                NAG_tol_train,
                verbose=verbose,
                x_init=x_init,
                return_stats=True,
            )
            optimizer.zero_grad()
            loss = lambda x_in: torch.sum(
                ((x_in - x) ** 2).view(x.shape[0], -1), -1
            ).mean()  # Defining the upper-level loss function
            if epoch == 0:
                loss_vals.append(loss(x_recon).item())
                psnr_vals.append(psnr(x_recon, x).mean().item())
                x_init = x_recon.detach().clone()

            x_recon = x_recon.detach()

            if reg and (epoch % 5) == 1:
                jac_loss = reg_para * jac_pow_loss(x_recon)
                jac_loss.backward()

            x_recon = x_recon.requires_grad_(True)
            # Computing the gradient of the upper-level objective with respect to the input
            grad_loss = torch.autograd.grad(loss(x_recon), x_recon, create_graph=False)[
                0
            ].detach()
            # Computing the approximate inverse Hessian vector product using CG/MINRES
            q = minres(
                lambda input: hessian_vector_product(
                    x_recon.detach(),
                    input,
                    data_fidelity,
                    y,
                    optimizer.regularizer,
                    lmbd,
                    physics_train,
                ),
                grad_loss,
                max_iter=cg_max_iter,
                tol=CG_tol,
            )
            # Computing the approximate hypergradient using the Jacobian vector product
            regularizer = jac_vector_product(
                x_recon, q/q.shape[0], data_fidelity, y, optimizer.regularizer, lmbd, physics_train
            )

            def closure(
                upper_grad_loss,
                loss,
                x_old,
                y,
                physics,
                data_fidelity,
                lmbd,
                optimizer,
                NAG_step_size,
                NAG_max_iter,
                rho_maid,
                eps,
                eps_old,
                max_line_search,
                fixed_eps,
                fixed_lr,
                verbose,
                eta=1e-4,
            ):
                """
                Closure function for MAID optimizer
                """
                if fixed_eps:
                    eps = eps_old
                if fixed_lr:
                    rho_maid = 1
                success_flag = False
                norm_loss_grad = lambda x_in: torch.norm(
                    upper_grad_loss(x_in.requires_grad_(True))
                )
                old_params = [
                    param.detach().clone()
                    for param in optimizer.regularizer.parameters()
                    if param.grad is not None
                ]
                old_grads = [
                    param.grad.detach().clone()
                    for param in optimizer.regularizer.parameters()
                    if param.grad is not None
                ]

                def revert(optimizer, params_before, grads_before):
                    """
                    This function reverts the optimizer's parameters and gradients to their state before taking a gradient step.
                    """
                    with torch.no_grad():
                        for param, p_old, g_old in zip(
                            [params for params in optimizer.regularizer.parameters() if params.grad is not None],
                            params_before,
                            grads_before,
                        ):
                            if param.grad is not None:
                                param.data.copy_(p_old)
                                param.grad.copy_(g_old)
                    if isinstance(optimizer, AdaGrad):
                        # pop the most recent entry from grad_sq_window when the line search fails
                        for param in optimizer.regularizer.parameters():
                            if hasattr(optimizer, "_grad_squared_accum"):
                                if param.name in optimizer._grad_squared_accum:
                                    if optimizer.window_size == 0:
                                        optimizer._grad_squared_accum[param.name].sub_(
                                            param.grad.pow(2)
                                        )
                                    else:
                                        optimizer._grad_squared_accum[param.name].pop()
                    return optimizer

                old_step = optimizer.lr
                for i in range(max_line_search):
                    optimizer.lr = (
                        optimizer.lr * rho_maid**i
                    )  # \rho_maid is the decay factor of line search
                    lr = optimizer.lr
                    grad_params = [
                        param.grad
                        for param in optimizer.regularizer.parameters()
                        if param.grad is not None
                    ]
                    optimizer.forward()  # \theta_{k+1} = \theta_k - lr * hypergrad
                    norm_grad_sq = 0.0  # Used in Adagrad to compute \|hypergrad\|^2_A where A is the AdaGrad preconditioner
                    if isinstance(optimizer, AdaGrad):
                        for name, param in optimizer.regularizer.named_parameters():
                            if param.grad is None:
                                continue
                            grad = param.grad.detach()

                            if name not in optimizer._grad_squared_accum:
                                print(
                                    "AdaGrad optimizer state missing '_grad_squared_accum'."
                                )
                                continue

                            state = optimizer._grad_squared_accum[name]

                            # Handle both full and truncated AdaGrad
                            if isinstance(state, torch.Tensor):
                                denom = state.sqrt() + optimizer.eps
                            elif isinstance(state, deque):
                                if len(state) == 0:
                                    continue  # avoid division by zero or empty window
                                avg_sq = torch.stack(list(state)).mean(dim=0)
                                denom = avg_sq.sqrt() + optimizer.eps
                            else:
                                raise TypeError(
                                    f"Unexpected type for _grad_squared_accum[{name}]: {type(state)}"
                                )

                            norm_grad_sq += (
                                (grad / denom) ** 2
                            ).sum()  # computes \|hypergrad\|^2_A where A is the AdaGrad preconditioner
                    hypergrad = torch.cat([g.reshape(-1) for g in grad_params])
                    if norm_grad_sq == 0.0:
                        if verbose:
                            print("norm hypergrad: ", torch.norm(hypergrad).item())
                    else:
                        if verbose:
                            print("norm hypergrad: ", torch.sqrt(norm_grad_sq).item())
                    x_new, stats = reconstruct_nmAPG(
                        y,
                        physics,
                        data_fidelity,
                        optimizer.regularizer,
                        lmbd,
                        NAG_step_size,
                        NAG_max_iter,
                        eps,
                        verbose=verbose,
                        x_init=x_old,
                        return_stats=True,
                    )
                    if verbose:
                        print(
                            "loss",
                            loss(x_new).item(),
                            "loss_old",
                            loss_vals[-1],
                            f"eps: {eps}",
                            f"lr: {lr}",
                            f"iter: {i}",
                        )
                    # Compute the line search condition
                    if lr * eta * torch.norm(hypergrad) ** 2 is None or torch.isnan(
                        lr * eta * torch.norm(hypergrad) ** 2
                    ):
                        line_search_RHS = 1e-7  # For numerical stability
                    else:
                        if norm_grad_sq == 0.0:
                            line_search_RHS = (
                                lr * eta * torch.norm(hypergrad) ** 2
                            )  # this is the case when AdaGrad is not used
                        else:
                            line_search_RHS = lr * eta * norm_grad_sq
                    line_search_LHS = (
                        loss(x_new)
                        - loss_vals[-1]
                        + norm_loss_grad(x_old) * eps
                        + norm_loss_grad(x_new) * eps_old
                        + 0.5 * eps**2
                        + 0.5 * eps_old**2
                    )
                    if verbose:
                        print(
                            "line_search_LHS: ",
                            line_search_LHS.item(),
                            "line_search_RHS: ",
                            line_search_RHS.item(),
                            "eps: ",
                            eps,
                        )
                    if (
                        line_search_LHS <= -line_search_RHS
                    ):  # checking the line search condition
                        with torch.no_grad():
                            x_old.copy_(x_new)
                        loss_old = loss(x_new)
                        loss_vals.append(loss(x_new).item())
                        psnr_vals.append(psnr(x_new, x).mean().item())
                        success_flag = True
                        return loss(x_new), x_new.detach(), success_flag, optimizer
                    optimizer = revert(optimizer, old_params, old_grads)
                    optimizer.lr = old_step
                    loss_old = loss_vals[-1]
                optimizer.zero_grad()
                return loss_old, x_old.detach(), success_flag, optimizer

            grad_loss = lambda x_in: torch.autograd.grad(
                loss(x_in), x_in, create_graph=False
            )[0].detach()
            loss = lambda x_in: (
                torch.sum(((x_in - x) ** 2).view(x.shape[0], -1), -1)
            ).mean()
            # Call the closure function to perform a MAID step
            _, x_recon, success, optimizer = closure(
                grad_loss,
                loss,
                x_init,
                y,
                physics_train,
                data_fidelity,
                lmbd,
                optimizer,
                NAG_step_size,
                NAG_max_iter,
                rho_maid,
                eps,
                eps_old,
                max_line_search,
                fixed_eps,
                fixed_lr,
                verbose,
            )
            if not success:
                print("Line search failed")
                eps_old = eps
                NAG_tol_train = NAG_tol_train * nu_under
                eps = NAG_tol_train
                CG_tol = CG_tol * nu_under
                max_line_search += 1
            else:
                eps_old = eps
                NAG_tol_train = NAG_tol_train * nu_over
                eps = NAG_tol_train
                CG_tol = CG_tol * nu_over
                max_line_search = 10
                optimizer.lr *= rho_over
            
            if not success:
                if hasattr(optimizer, "clear_memory"):
                    optimizer.clear_memory()
        
        print_str = f"[Epoch {epoch+1}] Train Loss: {loss_vals[-1]:.2E}, PSNR: {psnr_vals[-1]:.2f}"
        print(print_str)
        if logger is not None:
            logger.info(print_str)
        
        #if val_checkpoint is None or (epoch in val_checkpoint):
        if (epoch + 1) % validation_epochs == 0:
            regularizer.eval()
            with torch.no_grad():
                loss_vals_val = []
                psnr_vals_val = []
                for x_val in tqdm(val_dataloader):
                    x_val = preprocess(x_val, device)
                    y = physics(x_val)
                    x_init_val = y
                    x_recon_val = reconstruct_nmAPG(
                        y,
                        physics,
                        data_fidelity,
                        optimizer.regularizer,
                        lmbd,
                        NAG_step_size,
                        NAG_max_iter,
                        NAG_tol_val,
                        verbose=verbose,
                        x_init=x_init_val,
                    )
                    loss_validation = lambda x_in: torch.sum(
                        ((x_in - x_val) ** 2).view(x_val.shape[0], -1), -1
                    ).mean()
                    loss_vals_val.append(loss_validation(x_recon_val).item())
                    psnr_vals_val.append(psnr(x_recon_val, x_val).mean().item())
                mean_psnr_val = np.mean(psnr_vals_val)
                mean_loss_val = np.mean(loss_vals_val)
                loss_val.append(mean_loss_val)
                psnr_val.append(mean_psnr_val)
                
                print_str = f"[Epoch {epoch+1}] Val Loss: {mean_loss_val:.2E}, PSNR: {mean_psnr_val:.2f}"
                print(print_str)

                if logger is not None:
                    logger.info(print_str)

                # save checkpoint if the validation loss is lower than the previous one
                if mean_psnr_val > best_val_psnr:
                    best_val_psnr = mean_psnr_val
                    best_regularizer_state = copy.deepcopy(optimizer.regularizer.state_dict())
                    
        if NAG_tol_train < stopping_criterion or optimizer.lr < 1e-10:
            print(
                "Stopping criterion reached in epoch {0}: {1:.2E}".format(
                    epoch + 1, NAG_tol_train
                )
            )
            regularizer.load_state_dict(best_regularizer_state)
            break
    
    regularizer.load_state_dict(best_regularizer_state)
    
    return (
        optimizer.regularizer,
        loss_train,
        loss_val,
        psnr_train,
        psnr_val,
        eps,
        optimizer.lr,
        optimizer,
    )