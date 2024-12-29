import torch
import numpy as np
from tqdm import tqdm
from evaluation import reconstruct_NAG


def simple_bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=20,
    NAG_step_size=1e-2,
    NAG_max_iter=50,
    cg_max_iter=50,
    NAG_tol=1e-3,
    CG_tol=1e-6,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
    optimizer="AdamW",
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
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(regularizer.parameters(), lr=lr)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(regularizer.parameters(), lr=lr)
    elif optimizer == "SGD_Momentum":
        optimizer = torch.optim.SGD(regularizer.parameters(), lr=lr, momentum=0.9)
    elif optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(regularizer.parameters(), lr=lr, momentum=0.9)
    # TODO: Implement MAID optimizer
    # elif optimizer == "MAID":
    #     optimizer = MAID(regularizer.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        loss_vals = []
        for x in tqdm(train_dataloader):
            if device == "mps":
                x = x.to(torch.float32).to(device)
            else:
                x = x.to(device).to(torch.float)
            y = physics(x)
            # Solve the lower-level problem using NAG
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
            )
            optimizer.zero_grad()
            loss = lambda x_in: torch.sum(((x_in - x) ** 2).view(x.shape[0], -1), -1)
            loss_vals.append(loss(x_recon).mean().item())
            x_recon = x_recon.requires_grad_(True)
            # Computing the gradient of the upper-level objective with respect to the input
            grad_loss = torch.autograd.grad(
                loss(x_recon).mean(), x_recon, create_graph=True
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
            optimizer.step()
        print(
            "Average training loss in epoch {0}: {1:.2E}".format(
                epoch + 1, np.mean(loss_vals)
            )
        )
        if True:
            loss_vals = []
            for x in tqdm(val_dataloader):
                if device == "mps":
                    x = x.to(torch.float32).to(device)
                else:
                    x = x.to(device).to(torch.float)
                y = physics(x)
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
                )
                loss = lambda x_in: torch.sum(
                    ((x_in - x) ** 2).view(x.shape[0], -1), -1
                )
                loss_vals.append(loss(x_recon).mean().item())
            print(
                "Average validation loss in epoch {0}: {1:.2E}".format(
                    epoch + 1, np.mean(loss_vals)
                )
            )
