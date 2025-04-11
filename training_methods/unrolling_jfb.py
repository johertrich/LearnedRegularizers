import torch
import numpy as np
from tqdm import tqdm
from deepinv.loss.metric import PSNR
from evaluation import reconstruct_nmAPG


def unrolling_jfb(
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
    NAG_tol_val=1e-4,
    jfb_steps=1,
    jfb_step_size_factor=1.0,
    lr=0.005,
    lr_decay=0.99,
    device="cuda" if torch.cuda.is_available() else "cpu",
    verbose=False,
    validation_epochs=10,
    upper_loss=lambda x, y: torch.sum(((x - y) ** 2).view(x.shape[0], -1), -1),
):

    # Initialize optimizer for the upper-level
    optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr, weight_decay=1e-3)
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

            x_recon, L = reconstruct_nmAPG(
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
                return_L=True,
            )

            optimizer.zero_grad()
            x_recon = x_recon.detach()
            for _ in range(jfb_steps):
                grad = data_fidelity.grad(x_recon, y, physics) + regularizer.grad(
                    x_recon
                )
                x_recon = x_recon - jfb_step_size_factor / L * grad
            loss = upper_loss(x_recon, x).mean()
            loss.backward()
            loss_train_epoch += loss.item()
            psnr_train_epoch += psnr(x_recon, x).mean().item()
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
