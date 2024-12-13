import torch
import numpy as np
from tqdm import tqdm
from evaluation import reconstruct_NAG


def simple_unrolling_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=5,
    unrolling_steps=5,
    NAG_step_size=1e-2,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    optimizer = torch.optim.Adam(regularizer.parameters(), lr=lr)
    for epoch in range(epochs):
        loss_vals = []
        for x in tqdm(train_dataloader):
            x = x.to(device).to(torch.float)
            y = physics(x)
            x_recon = reconstruct_NAG(
                y,
                physics,
                data_fidelity,
                regularizer,
                lmbd,
                NAG_step_size,
                unrolling_steps,
                0,
            )
            loss = torch.sum(((x - x_recon) ** 2).view(x.shape[0], -1), -1)
            mean_loss = torch.mean(loss)
            optimizer.zero_grad()
            mean_loss.backward()
            optimizer.step()
            loss_vals.append(mean_loss.item())
        print(
            "Average training loss in epoch {0}: {1:.2E}".format(
                epoch + 1, np.mean(loss_vals)
            )
        )

        if True:
            loss_vals = []
            for x in tqdm(val_dataloader):
                x = x.to(device).to(torch.float)
                y = physics(x)
                x_recon = reconstruct_NAG(
                    y,
                    physics,
                    data_fidelity,
                    regularizer,
                    lmbd,
                    NAG_step_size,
                    unrolling_steps,
                    0,
                )
                loss = torch.sum(((x - x_recon) ** 2).view(x.shape[0], -1), -1)
                mean_loss = torch.mean(loss)
                loss_vals.append(mean_loss.item())
            print(
                "Average validation loss in epoch {0}: {1:.2E}".format(
                    epoch + 1, np.mean(loss_vals)
                )
            )
