import torch
import numpy as np
from tqdm import tqdm
from deepinv.loss.metric import PSNR
from torchvision.transforms import RandomCrop, Compose
from torch.utils.data import RandomSampler
from deepinv.optim.utils import conjugate_gradient
from evaluation import reconstruct_nmAPG
import random
from collections import deque
from torch.optim import Optimizer

class TruncatedAdaGrad(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-6, window_size=5):
        defaults = dict(lr=lr, eps=eps, window_size=window_size)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure else None
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                
                if 'grad_sq_window' not in state:
                    state['grad_sq_window'] = deque(maxlen=group['window_size'])

                # Append current squared gradient
                state['grad_sq_window'].append(grad.pow(2))

                # Compute mean of gradient memory
                grad_sq_avg = torch.stack(list(state['grad_sq_window'])).mean(dim=0)

                adjusted_lr = group['lr'] / (grad_sq_avg.sqrt() + group['eps'])
                p.data.add_(-adjusted_lr * grad)

        return loss

def simple_bilevel_training_maid(
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
    if precondition:
        optimizer = TruncatedAdaGrad(
            regularizer.parameters(),
            lr=lr,
            window_size=10
        )
    else:
        optimizer = torch.optim.SGD(regularizer.parameters(), lr=lr)
    success = False

    psnr = PSNR()

    # Training loop
    loss_vals = []
    psnr_vals = []
    loss_train = []
    loss_val = []
    psnr_train = []
    psnr_val = []

    if logs is None: # Initialize logs if not provided else continue with existing logs
        logs = {
            "train_loss": [],
            "psnr": [],
            "val_loss": [],
            "val_psnr": [],
            "eps": [],
            "lr": [],
            "grad_norm": [],
        }
    # Parameters for the MAID optimizer
    rho_maid = lr_decay
    nu_over = 1.25
    nu_under = 0.5
    rho_over = 1.25
    eps = NAG_tol_train
    eps_old = NAG_tol_train
    max_line_search = 10
    fixed_eps = False
    fixed_lr = False

    # initialize logs
    logs["eps"].append(eps)
    logs["lr"].append(optimizer.param_groups[0]["lr"])
    for img in train_dataloader:
        if device == "mps":
            img = img.to(torch.float32).to(device)
        else:
            img = img.to(device).to(torch.float)
        y = physics(img)
        loss_vals.append(
            torch.sum(((y - img) ** 2).view(img.shape[0], -1), -1).mean().item()
        )
        psnr_vals.append(psnr(y, img).mean().item())

    logs["train_loss"].append(loss_vals[-1])
    logs["psnr"].append(psnr_vals[-1])
    loss_vals_val = []
    psnr_vals_val = []
    for x_val in tqdm(val_dataloader):
        if device == "mps":
            x_val = x_val.to(torch.float32).to(device)
        else:
            x_val = x_val.to(device).to(torch.float)
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
            if device == "mps":
                x = x.to(torch.float32).to(device)
            else:
                x = x.to(device).to(torch.float)
            y = physics(x)

            if epoch == 0:
                x_init = y.detach().clone()

            else:
                # Warm start with the previous reconstruction for MAID
                if success:
                    x_init = x_recon.detach().clone()

            # if not success:
            x_recon = reconstruct_nmAPG(
                y,
                physics,
                data_fidelity,
                regularizer,
                lmbd,
                NAG_step_size,
                NAG_max_iter,
                NAG_tol_train,
                verbose=verbose,
                x_init=x_init,
            )
            optimizer.zero_grad()
            loss = lambda x_in: torch.sum(
                ((x_in - x) ** 2).view(x.shape[0], -1), -1
            ).mean()
            if epoch == 0:
                loss_vals.append(loss(x_recon).item())
                psnr_vals.append(psnr(x_recon, x).mean().item())
                x_init = x_recon.detach().clone()
            x_recon = x_recon.requires_grad_(True)
            # Computing the gradient of the upper-level objective with respect to the input
            grad_loss = torch.autograd.grad(loss(x_recon), x_recon, create_graph=False)[
                0
            ].detach()
            # Computing the approximate inverse Hessian vector product using CG
            q = conjugate_gradient(
                lambda input: hessian_vector_product(
                    x_recon.detach(),
                    input,
                    data_fidelity,
                    y,
                    regularizer,
                    lmbd,
                    physics,
                ),
                grad_loss,
                max_iter=cg_max_iter,
                tol=CG_tol,
            )
            # Computing the approximate hypergradient using the Jacobian vector product
            regularizer = jac_vector_product(
                x_recon, q, data_fidelity, y, regularizer, lmbd, physics
            )

            def closure(
                upper_grad_loss,
                loss,
                x_old,
                y,
                physics,
                data_fidelity,
                regularizer,
                lmbd,
                NAG_step_size,
                NAG_max_iter,
                NAG_tol_train,
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
                    for param in optimizer.param_groups[0]["params"]
                ]
                old_grads = [
                    param.grad.detach().clone()
                    for param in optimizer.param_groups[0]["params"]
                ]

                def revert(params_before, grads_before):
                    with torch.no_grad():
                        for i, param in enumerate(optimizer.param_groups[0]["params"]):
                            if param.grad is not None:
                                param.data.copy_(params_before[i])
                                param.grad.copy_(grads_before[i])
                        for i, param in enumerate(regularizer.parameters()):
                            if param.grad is not None:
                                param.data.copy_(params_before[i])
                                param.grad.copy_(grads_before[i])

                old_step = optimizer.param_groups[0]["lr"]
                for i in range(max_line_search):
                    optimizer.param_groups[0]["lr"] = (
                        optimizer.param_groups[0]["lr"] * rho_maid**i
                    )
                    lr = optimizer.param_groups[0]["lr"]
                    grad_params = [
                        param.grad
                        for param in optimizer.param_groups[0]["params"]
                        if param.grad is not None
                    ]
                    hypergrad = torch.cat([g.reshape(-1) for g in grad_params])
                    if verbose:
                        print("norm hypergrad: ", torch.norm(hypergrad).item())
                    logs["grad_norm"].append(torch.norm(hypergrad).item())
                    optimizer.step()
                    x_new = reconstruct_nmAPG(
                        y,
                        physics,
                        data_fidelity,
                        regularizer,
                        lmbd,
                        NAG_step_size,
                        NAG_max_iter,
                        eps,
                        verbose=verbose,
                        x_init=x_old,
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
                    if lr * eta * torch.norm(hypergrad) ** 2 is None or torch.isnan(
                        lr * eta * torch.norm(hypergrad) ** 2
                    ):
                        line_search_RHS = 1e-7
                    else:
                        line_search_RHS = lr * eta * torch.norm(hypergrad) ** 2
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
                    if line_search_LHS <= -line_search_RHS:
                        with torch.no_grad():
                            x_old.copy_(x_new)
                        loss_old = loss(x_new)
                        loss_vals.append(loss(x_new).item())
                        psnr_vals.append(psnr(x_new, x).mean().item())
                        logs["train_loss"].append(loss_vals[-1])
                        logs["psnr"].append(psnr_vals[-1])
                        success_flag = True
                        return loss(x_new), x_new.detach(), success_flag, regularizer
                    revert(old_params, old_grads)
                    optimizer.param_groups[0]["lr"] = old_step
                    loss_old = loss_vals[-1]

                    with torch.no_grad():
                        for param_reg, param in zip(
                            regularizer.parameters(),
                            optimizer.param_groups[0]["params"],
                        ):
                            param_reg.data.copy_(param)
                            param_reg.grad.copy_(param.grad)
                return loss_old, x_old.detach(), success_flag, regularizer

            grad_loss = lambda x_in: torch.autograd.grad(
                loss(x_in), x_in, create_graph=False
            )[0].detach()
            loss = lambda x_in: (
                torch.sum(((x_in - x) ** 2).view(x.shape[0], -1), -1)
            ).mean()
            _, x_recon, success, regularizer = closure(
                grad_loss,
                loss,
                x_init,
                y,
                physics,
                data_fidelity,
                regularizer,
                lmbd,
                NAG_step_size,
                NAG_max_iter,
                NAG_tol_train,
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
                optimizer.param_groups[0]["lr"] = (
                    optimizer.param_groups[0]["lr"] * rho_over
                )
            logs["eps"].append(eps)
            logs["lr"].append(optimizer.param_groups[0]["lr"])
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
                if device == "mps":
                    x_val = x_val.to(torch.float32).to(device)
                else:
                    x_val = x_val.to(device).to(torch.float)
                y = physics(x_val)
                x_init_val = y
                x_recon_val = reconstruct_nmAPG(
                    y,
                    physics,
                    data_fidelity,
                    regularizer,
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
                torch.save(regularizer.state_dict(), f"weights/MAID_{save_dir}.pt")
            loss_val.append(mean_loss_val)
            psnr_val.append(mean_psnr_val)
            logs["val_loss"].append(mean_loss_val)
            logs["val_psnr"].append(mean_psnr_val)
        torch.save(logs, f"weights/logs_{save_dir}.pt")
        torch.save(regularizer.state_dict(), f"weights/MAID_last_{save_dir}.pt")

        if NAG_tol_train < stopping_criterion or optimizer.param_groups[0]["lr"] < 1e-7:
            print(
                "Stopping criterion reached in epoch {0}: {1:.2E}".format(
                    epoch + 1, NAG_tol_train
                )
            )
            break
    return regularizer, loss_train, loss_val, psnr_train, psnr_val, eps, optimizer.param_groups[0]["lr"], logs
