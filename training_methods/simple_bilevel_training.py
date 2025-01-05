import torch
import numpy as np
from tqdm import tqdm
from evaluation import reconstruct_NAG
import matplotlib.pyplot as plt
from deepinv.loss.metric import PSNR
from torchvision.transforms import RandomCrop, Compose
from torch.utils.data import RandomSampler


def simple_bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=50,
    NAG_step_size=1e-2,
    NAG_max_iter=100,
    cg_max_iter=100,
    NAG_tol=1e-3,
    CG_tol=1e-8,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
    optimizer_alg="MAID",
):
    """
    Bilevel learning using inexact hypegradient methods described in
    https://academic.oup.com/imamat/article/89/1/254/7456287
    https://arxiv.org/abs/2308.10098
    https://arxiv.org/abs/2412.12049
    """

    def cg(A, b, x0, tol, max_iter, verbose=False):
        """
        Conjugate Gradient method to solve linear systems Ax = b
        """
        x = x0
        r = b - A(x)
        p = r
        rsold = torch.norm(r.view(r.size(0), -1), dim=1) ** 2
        for i in range(max_iter):
            Ap = A(p)
            alpha = rsold / torch.sum(
                p.view(p.size(0), -1) * Ap.view(Ap.size(0), -1), dim=1
            )
            alpha = alpha.view(-1, 1, 1, 1)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.norm(r.view(r.size(0), -1), dim=1) ** 2

            if torch.sqrt(torch.max(rsnew)) < tol * x.shape[0]:
                break

            p = r + (rsnew / rsold).view(-1, 1, 1, 1) * p
            rsold = rsnew
        if verbose:
            print(
                "CG iterations: ",
                i + 1,
                "max residual: ",
                torch.sqrt(torch.max(rsnew)).item(),
            )
        return x.detach()

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
    if optimizer_alg == "Adam":
        optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    elif optimizer_alg == "AdamW":
        optimizer = torch.optim.AdamW(regularizer.parameters(), lr=lr)
    elif optimizer_alg == "ISGD":
        optimizer = torch.optim.SGD(regularizer.parameters(), lr=lr)
    elif optimizer_alg == "ISGD_Momentum":
        optimizer = torch.optim.SGD(regularizer.parameters(), lr=lr, momentum=0.9)
    elif optimizer_alg == "RMSprop":
        optimizer = torch.optim.RMSprop(regularizer.parameters(), lr=lr, momentum=0.9)
    elif optimizer_alg == "MAID":
        optimizer = torch.optim.SGD(regularizer.parameters(), lr=lr, momentum=0.9)
        success = False

    # Training loop
    loss_vals = []
    if optimizer_alg == "MAID":
        # Parameters for the MAID optimizer
        rho = 0.1
        nu_over = 1.1
        nu_under = 0.5
        eps = NAG_tol
        eps_old = NAG_tol
        max_line_search = 3
        fixed_eps = False
        fixed_lr = False
    for epoch in range(epochs):
        if optimizer_alg != "MAID":
            loss_vals = []
        else:
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

            if epoch == 0 or optimizer_alg != "MAID":
                x_init = y
            else:
                # Warm start with the previous reconstruction for MAID
                x_init = x_recon
            x_recon = reconstruct_NAG(
                y,
                physics,
                data_fidelity,
                regularizer,
                lmbd,
                NAG_step_size,
                NAG_max_iter,
                NAG_tol,
                detach_grads=True,
                verbose=verbose,
                x_init=x_init,
            )
            optimizer.zero_grad()
            loss = lambda x_in: torch.sum(
                ((x_in - x) ** 2).view(x.shape[0], -1), -1
            ).mean()
            loss_vals.append(loss(x_recon).item())
            x_recon = x_recon.requires_grad_(True)
            # Computing the gradient of the upper-level objective with respect to the input
            grad_loss = torch.autograd.grad(
                loss(x_recon).mean(), x_recon, create_graph=False
            )[0].detach()
            # Computing the approximate inverse Hessian vector product using CG
            q = cg(
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
                torch.zeros_like(x_recon),
                CG_tol,
                cg_max_iter,
                verbose,
            )
            # Computing the approximate hypergradient using the Jacobian vector product
            regularizer = jac_vector_product(
                x_recon, q, data_fidelity, y, regularizer, lmbd, physics
            )
            for param, reg in zip(
                optimizer.param_groups[0]["params"], regularizer.parameters()
            ):
                if param.grad is not None:
                    param.grad = reg.grad
            if optimizer_alg != "MAID":
                optimizer.step()
            else:
                if len(loss_vals) > 1:
                    loss_vals.pop()

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
                    NAG_tol,
                    rho,
                    eps,
                    eps_old,
                    max_line_search,
                    fixed_eps,
                    fixed_lr,
                    verbose,
                    eta=1e-5,
                ):
                    """
                    Closure function for MAID optimizer
                    """
                    if fixed_eps:
                        eps = eps_old
                    if fixed_lr:
                        rho = 1
                    success_flag = False
                    norm_loss_grad = lambda x_in: torch.norm(
                        upper_grad_loss(x_in.requires_grad_(True)),
                        dim=(-2, -1),
                        keepdim=True,
                    ).mean()
                    old_params = [
                        param.clone()
                        for param in optimizer.param_groups[0]["params"]
                        if param.grad is not None
                    ]

                    def revert(params_before):
                        # Revert to the old parameters if condition not met
                        for i, param in enumerate(optimizer.param_groups[0]["params"]):
                            if param.grad is not None:
                                param.data = params_before[i].data

                    old_step = optimizer.param_groups[0]["lr"]
                    for i in range(max_line_search):
                        optimizer.param_groups[0]["lr"] = (
                            optimizer.param_groups[0]["lr"] * rho**i
                        )
                        lr = optimizer.param_groups[0]["lr"]
                        grad_params = [
                            param.grad
                            for param in optimizer.param_groups[0]["params"]
                            if param.grad is not None
                        ]
                        hypergrad = torch.cat([g.reshape(-1) for g in grad_params])
                        # normalise the hypergradient by the norm of the hypergradient
                        for _, param in enumerate(optimizer.param_groups[0]["params"]):
                            if param.grad is not None:
                                param.grad = param.grad / torch.norm(hypergrad)
                        optimizer.step()
                        x_new = reconstruct_NAG(
                            y,
                            physics,
                            data_fidelity,
                            regularizer,
                            lmbd,
                            NAG_step_size,
                            NAG_max_iter,
                            NAG_tol,
                            detach_grads=True,
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
                        if (
                            loss(x_new)
                            - loss_vals[-1]
                            + norm_loss_grad(x_old) * eps_old
                            + norm_loss_grad(x_new) * eps
                            <= -lr * eta**2 * torch.norm(hypergrad) ** 2
                        ):
                            x_old = x_new
                            loss_old = loss(x_new)
                            if loss(x_new) < loss_vals[-1]:
                                loss_vals.append(loss(x_new).item())
                            success_flag = True
                            return loss(x_new), x_new, success_flag, regularizer
                        revert(old_params)
                        optimizer.param_groups[0]["lr"] = old_step
                        loss_old = loss_vals[-1]
                    return loss_old, x_old, success_flag, regularizer

                grad_loss = lambda x_in: torch.autograd.grad(
                    loss(x_in).mean(), x_in, create_graph=False
                )[0].detach()
                loss = lambda x_in: (
                    torch.sum(((x_in - x) ** 2).view(x.shape[0], -1), -1)
                ).mean()
                _, x_recon, success, regularizer = closure(
                    grad_loss,
                    loss,
                    x_recon,
                    y,
                    physics,
                    data_fidelity,
                    regularizer,
                    lmbd,
                    NAG_step_size,
                    NAG_max_iter,
                    NAG_tol,
                    rho,
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
                    NAG_tol = NAG_tol * nu_under
                    eps = NAG_tol
                    CG_tol = CG_tol * nu_under
                    max_line_search += 1
                else:
                    eps_old = eps
                    NAG_tol = NAG_tol * nu_over
                    eps = NAG_tol
                    CG_tol = CG_tol * nu_over
                    max_line_search = 5
                    optimizer.param_groups[0]["lr"] = (
                        optimizer.param_groups[0]["lr"] * nu_over
                    )
            psnr = PSNR()
            print(
                "Mean PSNR training {0:.2f}".format(psnr(x_recon, x).mean().item()),
                "loss",
                loss_vals[-1],
            )
        if optimizer_alg != "MAID":
            print(
                "Average training loss in epoch {0}: {1:.2E}".format(
                    epoch + 1, np.mean(loss_vals)
                )
            )

        # set regularizer parameters to the parameters of the trained regularizer
        for param, reg in zip(
            optimizer.param_groups[0]["params"], regularizer.parameters()
        ):
            if param.grad is not None:
                reg.data = torch.clone(param.data)
        # save the trained regularizer
        torch.save(regularizer.state_dict(), "weights/simple_ICNN_bilevel.pt")
        if True:
            loss_vals_val = []
            for x_val in tqdm(val_dataloader):
                if device == "mps":
                    x_val = x_val.to(torch.float32).to(device)
                else:
                    x_val = x_val.to(device).to(torch.float)
                y = physics(x_val)
                x_recon_val = reconstruct_NAG(
                    y,
                    physics,
                    data_fidelity,
                    regularizer,
                    lmbd,
                    NAG_step_size,
                    NAG_max_iter,
                    NAG_tol,
                    detach_grads=True,
                )
                loss_validation = lambda x_in: torch.sum(
                    ((x_in - x_val) ** 2).view(x_val.shape[0], -1), -1
                ).mean()
                loss_vals_val.append(loss_validation(x_recon_val).item())
            print(
                "Average validation loss in epoch {0}: {1:.2E}".format(
                    epoch + 1, np.mean(loss_vals_val)
                )
            )
    return regularizer
