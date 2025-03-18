import torch
import numpy as np
from tqdm import tqdm
from deepinv.loss.metric import PSNR
from deepinv.optim.utils import minres
from evaluation import reconstruct_NAG_LS, reconstruct_NAG_RS


def bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=100,
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-6,
    linesearch=True,
    minres_max_iter=1000,
    minres_tol=1e-6,
    lr=0.005,
    lr_decay=0.99,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
    validation_epochs=10,
    upper_loss=lambda x, y: torch.sum(((x - y) ** 2).view(x.shape[0], -1), -1),
):

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
    optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    psnr = PSNR()

    len_train = len(train_dataloader)
    len_val = len(val_dataloader)

    # Training loop
    loss_train = []
    loss_val = []
    psnr_train = []
    psnr_val = []
    for epoch in range(epochs):

        loss_train_epoch = 0
        psnr_train_epoch = 0

        for x in tqdm(train_dataloader):
            if device == "mps":
                x = x.to(torch.float32).to(device)
            else:
                x = x.to(device).to(torch.float)
            y = physics(x)

            x_init = y

            if linesearch:
                x_recon = reconstruct_NAG_LS(
                    y,
                    physics,
                    data_fidelity,
                    regularizer,
                    lmbd,
                    NAG_step_size,
                    NAG_max_iter,
                    NAG_tol_train,
                    rho=0.9,
                    delta=0.9,
                    verbose=verbose,
                    x_init=x_init,
                    progress=False,
                )
            else:
                # NAG_step_size = 1/torch.exp(regularizer.beta)
                x_recon = reconstruct_NAG_RS(
                    y,
                    physics,
                    data_fidelity,
                    regularizer,
                    lmbd,
                    NAG_step_size,
                    NAG_max_iter,
                    NAG_tol_train,
                    detach_grads=True,
                    verbose=verbose,
                    x_init=x_init,
                    progress=False,
                    restart=True,
                )

            optimizer.zero_grad()
            loss = lambda x_in: upper_loss(x, x_in).mean()
            loss_train_epoch += loss(x_recon).item()
            psnr_train_epoch += psnr(x_recon, x).mean().item()
            x_recon = x_recon.requires_grad_(True)
            # Computing the gradient of the upper-level objective with respect to the input
            grad_loss = torch.autograd.grad(loss(x_recon), x_recon, create_graph=False)[
                0
            ].detach()
            # Computing the approximate inverse Hessian vector product using CG
            q = minres(
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
                max_iter=minres_max_iter,
                tol=minres_tol,
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

        scheduler.step()
        mean_loss_train = loss_train_epoch / len_train
        mean_psnr_train = psnr_train_epoch / len_train

        print(
            "Mean PSNR training in epoch {0}: {1:.2f}".format(
                epoch + 1, mean_psnr_train
            )
        )
        print(
            "Mean loss training in epoch {0}: {1:.2E}".format(
                epoch + 1, mean_loss_train
            )
        )

        loss_train.append(mean_loss_train)
        psnr_train.append(mean_psnr_train)

        # set regularizer parameters to the parameters of the trained regularizer
        for param, reg in zip(
            optimizer.param_groups[0]["params"], regularizer.parameters()
        ):
            if param.grad is not None:
                reg.data = torch.clone(param.data)

        if (epoch + 1) % validation_epochs == 0:
            loss_val_epoch = 0
            psnr_val_epoch = 0
            for x_val in tqdm(val_dataloader):
                if device == "mps":
                    x_val = x_val.to(torch.float32).to(device)
                else:
                    x_val = x_val.to(device).to(torch.float)
                y = physics(x_val)
                x_init_val = y

                if linesearch:
                    x_recon_val = reconstruct_NAG_LS(
                        y,
                        physics,
                        data_fidelity,
                        regularizer,
                        lmbd,
                        NAG_step_size,
                        NAG_max_iter,
                        NAG_tol_val,
                        rho=0.9,
                        delta=0.9,
                        verbose=verbose,
                        x_init=x_init_val,
                        progress=False,
                    )
                else:
                    x_recon_val = reconstruct_NAG_RS(
                        y,
                        physics,
                        data_fidelity,
                        regularizer,
                        lmbd,
                        NAG_step_size,
                        NAG_max_iter,
                        NAG_tol_val,
                        detach_grads=True,
                        verbose=verbose,
                        x_init=x_init_val,
                        progress=False,
                        restart=True,
                    )

                loss_validation = lambda x_in: torch.sum(
                    ((x_in - x_val) ** 2).view(x_val.shape[0], -1), -1
                ).mean()
                loss_val_epoch += loss_validation(x_recon_val).item()
                psnr_val_epoch += psnr(x_recon_val, x_val).mean().item()

            mean_loss_val = loss_val_epoch / len_val
            mean_psnr_val = psnr_val_epoch / len_val

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
            loss_val.append(mean_loss_val)
            psnr_val.append(mean_psnr_val)
    return regularizer, loss_train, loss_val, psnr_train, psnr_val
