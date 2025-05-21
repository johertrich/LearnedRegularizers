import torch
import numpy as np
from tqdm import tqdm
from deepinv.loss.metric import PSNR
from evaluation import reconstruct_nmAPG
import copy


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
    upper_loss=lambda x, y: torch.sum(((x - y) ** 2).view(x.shape[0], -1), -1),
):

    optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    psnr = PSNR()

    loss_train = []
    loss_val = []
    psnr_train = []
    psnr_val = []

    best_val_psnr = -float("inf")
    best_regularizer_state = copy.deepcopy(regularizer.state_dict())

    for epoch in range(epochs):
        # ---- Training ----
        regularizer.train()
        train_loss_epoch = 0
        train_psnr_epoch = 0

        for x in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Train"):
            x = x.to(device).to(torch.float32)
            y = physics(x)
            x_noisy = physics.A_dagger(y)
            
            x_recon, L = reconstruct_nmAPG(
                y, physics, data_fidelity, regularizer, lmbd,
                NAG_step_size, NAG_max_iter, NAG_tol_train,
                verbose=verbose, x_init=x_noisy, return_L=True,
            )

            optimizer.zero_grad()
	        x_recon = x_recon.detach().requires_grad_(True)
            for _ in range(jfb_steps):
                grad = data_fidelity.grad(x_recon, y, physics) + regularizer.grad(
                    x_recon
                )
                x_recon = x_recon - jfb_step_size_factor / L * grad
            loss = upper_loss(x_recon, x).mean()
            loss.backward()
            train_loss_epoch += loss.item()
            train_psnr_epoch += psnr(x_recon, x).mean().item()

            optimizer.step()

        scheduler.step()
        mean_train_loss = train_loss_epoch / len(train_dataloader)
        mean_train_psnr = train_psnr_epoch / len(train_dataloader)
        loss_train.append(mean_train_loss)
        psnr_train.append(mean_train_psnr)

        print(f"[Epoch {epoch+1}] Train Loss: {mean_train_loss:.2E}, PSNR: {mean_train_psnr:.2f}")

        # ---- Validation ----
        regularizer.eval()
        with torch.no_grad():
            val_loss_epoch = 0
            val_psnr_epoch = 0
            for x_val in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Val"):
                x_val = x_val.to(device).to(torch.float32)
                y_val = physics(x_val)
                x_val_noisy = physics.A_dagger(y_val)

                x_recon_val = reconstruct_nmAPG(
                    y_val, physics, data_fidelity, regularizer, lmbd,
                    NAG_step_size, NAG_max_iter, NAG_tol_val,
                    verbose=verbose, x_init=x_val_noisy,
                )

                val_loss_epoch += upper_loss(x_val, x_recon_val).mean().item()
                val_psnr_epoch += psnr(x_recon_val, x_val).mean().item()

            mean_val_loss = val_loss_epoch / len(val_dataloader)
            mean_val_psnr = val_psnr_epoch / len(val_dataloader)
            loss_val.append(mean_val_loss)
            psnr_val.append(mean_val_psnr)

            print(f"[Epoch {epoch+1}] Val Loss: {mean_val_loss:.2E}, PSNR: {mean_val_psnr:.2f}")

            # ---- Save best regularizer based on validation PSNR ----
            if mean_val_psnr > best_val_psnr:
                best_val_psnr = mean_val_psnr
                best_regularizer_state = copy.deepcopy(regularizer.state_dict())

    # Load best regularizer
    regularizer.load_state_dict(best_regularizer_state)

    return regularizer, loss_train, loss_val, psnr_train, psnr_val
