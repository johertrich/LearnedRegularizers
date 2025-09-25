import torch
from tqdm import tqdm
from deepinv.loss.metric import PSNR
import copy
from .utils.adabelief import AdaBelief


def score_training(
    regularizer,
    train_dataloader,
    val_dataloader,
    sigma: float = 1e-2,
    epochs=100,
    lr=0.005,
    lr_decay=0.99,
    weight_decay=0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    validation_epochs=20,
    logger=None,
    dynamic_range_psnr=False,
    savestr=None,
    loss_fn=lambda x, y: torch.sum((x - y) ** 2),
    adabelief=False,
    model_selection=True,
):

    if adabelief:
        optimizer = AdaBelief(
            [
                {"params": regularizer.parameters(), "lr": lr},
            ],
            lr=lr,
            betas=(0.5, 0.9),
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            regularizer.parameters(), lr=lr, weight_decay=weight_decay
        )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    if dynamic_range_psnr:
        psnr = PSNR(max_pixel=None)
    else:
        psnr = PSNR()

    loss_train = []
    loss_val = []
    psnr_train = []
    psnr_val = []

    best_val_psnr = -float("inf")
    best_regularizer_state = copy.deepcopy(regularizer.state_dict())

    for epoch in range(epochs):
        regularizer.train()
        train_loss_epoch = 0
        train_psnr_epoch = 0

        for train_step, x in tqdm(
            enumerate(train_dataloader), desc=f"Epoch {epoch+1}/{epochs} - Train"
        ):
            x = x.to(device)
            noise = torch.randn_like(x)
            y = x + sigma * noise
            xhat = y - regularizer.grad(y)
            loss = loss_fn(xhat, x)
            optimizer.zero_grad()
            loss.backward()
            xhat = xhat.detach()

            optimizer.step()
            train_loss_epoch += loss.item()
            train_psnr_epoch += psnr(xhat, x).mean().item()

        scheduler.step()
        mean_train_loss = train_loss_epoch / len(train_dataloader)
        mean_train_psnr = train_psnr_epoch / len(train_dataloader)
        loss_train.append(mean_train_loss)
        psnr_train.append(mean_train_psnr)

        print_str = f"[Epoch {epoch+1}] Train Loss: {mean_train_loss:.2E}, PSNR: {mean_train_psnr:.2f}"
        print(print_str)
        if logger is not None:
            logger.info(print_str)

        if (epoch + 1) % validation_epochs == 0:
            regularizer.eval()
            with torch.no_grad():
                val_loss_epoch = 0
                val_psnr_epoch = 0
                for x_val in tqdm(
                    val_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Val"
                ):
                    x_val = x_val.to(device).to(torch.float32)
                    noise = torch.randn_like(x_val)
                    y = x_val + sigma * noise
                    xhat = y - regularizer.grad(y)
                    loss = loss_fn(xhat, x_val)

                    val_loss_epoch += loss.item()
                    val_psnr_epoch += psnr(xhat, x_val).mean().item()

                mean_val_loss = val_loss_epoch / len(val_dataloader)
                mean_val_psnr = val_psnr_epoch / len(val_dataloader)
                loss_val.append(mean_val_loss)
                psnr_val.append(mean_val_psnr)

                print_str = f"[Epoch {epoch+1}] Val Loss: {mean_val_loss:.2E}, PSNR: {mean_val_psnr:.2f}"
                print(print_str)

                if savestr is not None:
                    torch.save(
                        regularizer.state_dict(),
                        savestr + "_epoch_" + str(epoch) + ".pt",
                    )

                if logger is not None:
                    logger.info(print_str)

                # ---- Save best regularizer based on validation PSNR ----
                if mean_val_psnr > best_val_psnr and model_selection:
                    best_val_psnr = mean_val_psnr
                    best_regularizer_state = copy.deepcopy(regularizer.state_dict())

    # Load best regularizer
    if model_selection:
        regularizer.load_state_dict(best_regularizer_state)

    return regularizer, loss_train, loss_val, psnr_train, psnr_val
