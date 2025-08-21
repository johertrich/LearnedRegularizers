import torch
import numpy as np
from tqdm import tqdm
from deepinv.loss.metric import PSNR
from torchvision.transforms import RandomCrop, Compose
from torch.utils.data import RandomSampler
from deepinv.optim.utils import conjugate_gradient, minres
from evaluation import reconstruct_nmAPG
import random
from collections import deque
from torch.optim import Optimizer
import torch.nn as nn

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
        self.window_size = window_size  # Optional: for truncated AdaGrad. We do not use it by default
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
                        avg_sq_grad = torch.stack(list(self._grad_squared_accum[name])).mean(dim=0)
                        adjusted_lr = self.lr / (avg_sq_grad.sqrt() + self.eps)
                    else:
                        # Standard AdaGrad
                        self._grad_squared_accum[name].add_(grad.pow(2))
                        adjusted_lr = self.lr / (self._grad_squared_accum[name].sqrt() + self.eps)

                    param -= adjusted_lr * grad
def preprocess(x, device):
    dtype = torch.float32 if device == "mps" else torch.float
    return x.to(dtype).to(device)

def bilevel_training_maid(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=50,
    NAG_step_size=1e-2,
    NAG_max_iter=400,
    NAG_tol_train=1e-1,
    NAG_tol_val=1e-6,
    cg_max_iter=1000,
    CG_tol=1e-6,
    lr=1e-3,
    lr_decay=0.5,
    stopping_criterion=5e-5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    precondition=False,
    verbose=False,
    save_dir="",
    val_checkpoint=None,
    logs=None,
    optimizer=None,
    algorithm="AdaGrad",
):
    """
    Bilevel learning using inexact hypegradient methods described in
    https://academic.oup.com/imamat/article/89/1/254/7456287
    https://arxiv.org/abs/2308.10098
    https://arxiv.org/abs/2412.12049
    """

    def hessian_vector_product(x, v, data_fidelity, y, regularizer, lmbd, physics):
        """
        Compute the Hessian vector product of the lower level objective
        """
        x = x.requires_grad_(True)
        grad = data_fidelity.grad(x, y, physics) + lmbd * regularizer.grad(x)
        dot = torch.dot(grad.view(-1), v.view(-1))
        hvp = torch.autograd.grad(dot, x, create_graph=False)[0].detach()
        return hvp.detach()

    def jac_vector_product(x, v, data_fidelity, y, regularizer, lmbd, physics):
        """
        Compute the Jacobian (mixed derivative) vector product of the lower level objective
        and the final approximate hypergradient as the gradient of the regularizer parameters
        """
        grad_lower_level = lambda x: data_fidelity.grad(
            x, y, physics
        ) + lmbd * regularizer.grad(x)
        for param in regularizer.parameters():
            dot = torch.dot(grad_lower_level(x).view(-1), v.view(-1))
            param.grad = -torch.autograd.grad(dot, param, create_graph=False)[
                0
            ].detach()

        return regularizer

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
    success = False # Flag for backtracking line search success
    psnr = PSNR() # PSNR metric definition
    # logging lists and dictionaries
    loss_vals = []
    psnr_vals = []
    loss_train = []
    loss_val = []
    psnr_train = []
    psnr_val = []
    first_run = False
    if logs is None: # Initialize logs if not provided else continue with existing logs
        logs = {
            "train_loss": [],
            "psnr": [],
            "val_loss": [],
            "val_psnr": [],
            "eps": [],
            "lr": [],
            "grad_norm": [],
            "algorithm": algorithm,
        }
        first_run = True
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

    # initialize logs
    if first_run:
        logs["eps"].append(eps)
        logs["lr"].append(optimizer.lr)
        # Initialising training PSNR and loss values before training starts
        for img in train_dataloader:
            img = preprocess(img, device)
            y = physics(img)
            loss_vals.append(
                torch.sum(((y - img) ** 2).view(img.shape[0], -1), -1).mean().item()
            )
            psnr_vals.append(psnr(y, img).mean().item())

        logs["train_loss"].append(loss_vals[-1])
        logs["psnr"].append(psnr_vals[-1])
        loss_vals_val = []
        psnr_vals_val = []
        # Initialising validation PSNR and loss values before training starts
        for x_val in tqdm(val_dataloader):
            x_val = preprocess(x_val, device)
            y = physics(x_val)
            loss_validation = lambda x_in: torch.sum(
                ((x_in - x_val) ** 2).view(x_val.shape[0], -1), -1
            ).mean()
            loss_vals_val.append(loss_validation(y).item())
            psnr_vals_val.append(psnr(y, x_val).mean().item())

        mean_psnr_val = np.mean(psnr_vals_val)
        mean_loss_val = np.mean(loss_vals_val)
        logs["val_loss"].append(mean_loss_val)
        logs["val_psnr"].append(mean_psnr_val)
    # Training loop
    for epoch in range(epochs):
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
            y = physics(x)
            if epoch == 0:
                x_init = y.detach().clone()
            else:
                # Warm start with the previous reconstruction for MAID
                if success:
                    x_init = x_recon.detach().clone()
            # Solve the lower-level problem to compute the hypergradient
            x_recon, stats = reconstruct_nmAPG(
                y,
                physics,
                data_fidelity,
                optimizer.regularizer,
                lmbd,
                NAG_step_size,
                NAG_max_iter,
                NAG_tol_train,
                verbose=verbose,
                x_init=x_init,
                return_stats = True
            )
            optimizer.zero_grad()
            loss = lambda x_in: torch.sum(
                ((x_in - x) ** 2).view(x.shape[0], -1), -1
            ).mean() # Defining the upper-level loss function
            if epoch == 0:
                loss_vals.append(loss(x_recon).item())
                psnr_vals.append(psnr(x_recon, x).mean().item())
                x_init = x_recon.detach().clone()
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
                    physics,
                ),
                grad_loss,
                max_iter=cg_max_iter,
                tol=CG_tol,
            )
            # Computing the approximate hypergradient using the Jacobian vector product
            regularizer = jac_vector_product(
                x_recon, q, data_fidelity, y, optimizer.regularizer, lmbd, physics
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
                ]
                old_grads = [
                    param.grad.detach().clone()
                    for param in optimizer.regularizer.parameters()
                ]     
                def revert(optimizer, params_before, grads_before):
                    """
                    This function reverts the optimizer's parameters and gradients to their state before taking a gradient step. 
                    """
                    with torch.no_grad():
                        for param, p_old, g_old in zip(optimizer.regularizer.parameters(), params_before, grads_before):
                            if param.grad is not None:
                                param.data.copy_(p_old)
                                param.grad.copy_(g_old)
                    if isinstance(optimizer, AdaGrad):
                        # pop the most recent entry from grad_sq_window when the line search fails
                        for param in optimizer.regularizer.parameters():
                            if hasattr(optimizer, '_grad_squared_accum'):
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
                    optimizer.lr  = (
                        optimizer.lr * rho_maid**i
                    ) # \rho_maid is the decay factor of line search
                    lr = optimizer.lr
                    grad_params = [
                        param.grad
                        for param in optimizer.regularizer.parameters()
                        if param.grad is not None
                    ]
                    optimizer.forward() # \theta_{k+1} = \theta_k - lr * hypergrad
                    norm_grad_sq = 0.0 # Used in Adagrad to compute \|hypergrad\|^2_A where A is the AdaGrad preconditioner
                    if isinstance(optimizer, AdaGrad):
                        for name, param in optimizer.regularizer.named_parameters():
                            if param.grad is None:
                                continue
                            grad = param.grad.detach()

                            if name not in optimizer._grad_squared_accum:
                                print("AdaGrad optimizer state missing '_grad_squared_accum'.")
                                continue

                            state = optimizer._grad_squared_accum[name]

                            # Handle both full and truncated AdaGrad
                            if isinstance(state, torch.Tensor):
                                denom = (state.sqrt() + optimizer.eps)
                            elif isinstance(state, deque):
                                if len(state) == 0:
                                    continue  # avoid division by zero or empty window
                                avg_sq = torch.stack(list(state)).mean(dim=0)
                                denom = (avg_sq.sqrt() + optimizer.eps)
                            else:
                                raise TypeError(f"Unexpected type for _grad_squared_accum[{name}]: {type(state)}")

                            norm_grad_sq += ((grad / denom) ** 2).sum()    # computes \|hypergrad\|^2_A where A is the AdaGrad preconditioner      
                    hypergrad = torch.cat([g.reshape(-1) for g in grad_params])
                    logs["grad_norm"].append(torch.norm(hypergrad).item())
                    if norm_grad_sq == 0.0:
                        if verbose:
                            print("norm hypergrad: ", torch.norm(hypergrad).item())
                    else:
                        if verbose:
                            print(
                                "norm hypergrad: ",
                                torch.sqrt(norm_grad_sq).item()
                            )
                    x_new, stats= reconstruct_nmAPG(
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
                        return_stats = True
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
                        line_search_RHS = 1e-7 # For numerical stability
                    else:
                        if norm_grad_sq == 0.0:
                            line_search_RHS = lr * eta * torch.norm(hypergrad) ** 2 # this is the case when AdaGrad is not used
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
                    if line_search_LHS <= -line_search_RHS: # checking the line search condition
                        with torch.no_grad():
                            x_old.copy_(x_new)
                        loss_old = loss(x_new)
                        loss_vals.append(loss(x_new).item())
                        psnr_vals.append(psnr(x_new, x).mean().item())
                        logs["train_loss"].append(loss_vals[-1])
                        logs["psnr"].append(psnr_vals[-1])
                        success_flag = True
                        return loss(x_new), x_new.detach(), success_flag, optimizer
                    optimizer = revert(optimizer, old_params, old_grads)
                    optimizer.lr= old_step
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
            logs["eps"].append(eps)
            logs["lr"].append(optimizer.lr)
            if not success:
                if hasattr(optimizer, "clear_memory"):
                    optimizer.clear_memory()
        print(
            "Mean PSNR training in epoch {0}: {1:.2f}".format(epoch + 1, psnr_vals[-1])
        )
        print(
            "Mean loss training in epoch {0}: {1:.2E}".format(epoch + 1, loss_vals[-1])
        )
        if val_checkpoint is None or (epoch in val_checkpoint):
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
            print(
                "Average validation psnr in epoch {0}: {1:.2f}".format(
                    epoch + 1, mean_psnr_val
                )
            )
            print(
                "Average validation loss in epoch {0}: {1:.2E}".format(
                    epoch + 1, mean_loss_val
                )
            )
            # save checkpoint if the validation loss is lower than the previous one
            if len(logs["val_loss"]) > 1 and mean_loss_val < min(logs["val_loss"]):
                print(
                    "Validation loss improved from {0:.2E} to {1:.2E}".format(
                        min(logs["val_loss"]), mean_loss_val
                    )
                )
            loss_val.append(mean_loss_val)
            psnr_val.append(mean_psnr_val)
            logs["val_loss"].append(mean_loss_val)
            logs["val_psnr"].append(mean_psnr_val)
        torch.save(logs, f"weights/logs_{save_dir}.pt")
        # torch.save(regularizer.state_dict(), f"weights/MAID_{save_dir}.pt")

        if NAG_tol_train < stopping_criterion or optimizer.lr < 1e-7:
            print(
                "Stopping criterion reached in epoch {0}: {1:.2E}".format(
                    epoch + 1, NAG_tol_train
                )
            )
            break
    return optimizer.regularizer, loss_train, loss_val, psnr_train, psnr_val, eps, optimizer.lr, logs, optimizer