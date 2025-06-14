from priors import LSR
import torch
from training_methods import bilevel_training
from dataset import get_dataset
from torchvision.transforms import RandomCrop, CenterCrop, Compose
from torchvision.transforms import (
    RandomCrop,
    RandomVerticalFlip,
    Compose,
    RandomHorizontalFlip,
    CenterCrop,
    RandomApply,
    RandomRotation,
)
from operators import get_operator
import logging
import datetime

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "CT"

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="training_LSR_jfb_CT_" + str(datetime.datetime.now()) + ".log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

physics, data_fidelity = get_operator(problem, device)

train_dataset = get_dataset("LoDoPaB", test=False)
val_dataset = get_dataset("LoDoPaB", test=False)
# splitting in training and validation set
test_ratio = 0.1
test_len = int(len(train_dataset) * 0.1)
train_len = len(train_dataset) - test_len
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = get_dataset(
    "LoDoPaB", test=True
)  # evaluate during training, must not be used for model selection

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=8, shuffle=True, drop_last=True, num_workers=8
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True, num_workers=8
)

# define regularizer
lmbd = 1.0
regularizer = LSR(nc=[32, 64, 128, 256], pretrained_denoiser=False).to(device)
regularizer.sigma.data *= 0.5

params = 0
for p in regularizer.parameters():
    params += p.numel()
print(params)

# Pretraining
load_pretrain = True
if load_pretrain:
    regularizer.load_state_dict(torch.load("weights/LSR_pretraining_on_LoDoPaB.pt"),strict=False)
else:
    from tqdm import tqdm
    import numpy as np
    from deepinv.loss.metric import PSNR

    pretrain_dataset = get_dataset("LoDoPaB", test=False)
    pretrain_dataloader = torch.utils.data.DataLoader(
        pretrain_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=8
    )

    psnr = PSNR()
    epochs_pretraining = 50
    noise_level_range = [2.5 / 255.0, 20.0 / 255.0]
    optimizer = torch.optim.Adam(regularizer.model.parameters(), lr=1e-4)
    for p in regularizer.parameters():
        p.requires_grad_(True)
    regularizer.alpha.requires_grad_(False)
    regularizer.sigma.requires_grad_(False)
    for epoch in range(epochs_pretraining):
        losses = []
        for x in tqdm(
            pretrain_dataloader, desc=f"Epoch {epoch+1}/{epochs_pretraining} - Train"
        ):
            optimizer.zero_grad()
            x = x.to(device)
            noise_levels = (
                torch.rand((x.shape[0], 1, 1, 1), device=device) + noise_level_range[0]
            ) * (noise_level_range[1] - noise_level_range[0])
            noisy = x + noise_levels * torch.randn_like(x)
            pred = regularizer.model(noisy, noise_levels)
            loss = torch.sum(torch.abs(pred - x))
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        print(np.mean(losses))
        if (epoch + 1) % 1 == 0:
            print("Validation")
            losses = []
            psnrs = []
            for x in tqdm(
                val_dataloader, desc=f"Epoch {epoch+1}/{epochs_pretraining} - Val"
            ):
                x = x.to(device)
                noisy = x + 0.05 * torch.randn_like(x)
                pred = regularizer.model(noisy, 0.05)
                loss = torch.sum(torch.abs(pred - x))
                losses.append(loss.item())
                psnrs.append(psnr(pred, x).mean().item())
            print(np.mean(losses), np.mean(psnrs))
            logger.info(
                "Pretrain Validation Epoch {0} of {1}, loss {2:.2f}, PSNR {3:.2f}".format(
                    epoch + 1, epochs_pretraining, np.mean(losses), np.mean(psnrs)
                )
            )

    torch.save(regularizer.state_dict(), f"weights/LSR_pretraining_on_LoDoPaB.pt")

# Parameter fitting

load_fittet_parameters=False
if load_fittet_parameters:
    regularizer.load_state_dict(torch.load("weights/LSR_pretraining_and_parameter_fitting_on_LoDoPaB.pt"))
else:
    for p in regularizer.parameters():
        p.requires_grad_(False)
    regularizer.alpha.requires_grad_(True)
    regularizer.sigma.requires_grad_(False)
    regularizer.sigma.data=torch.tensor(-1.8).to(regularizer.sigma.data)
    regularizer.alpha.data=torch.tensor(5.8).to(regularizer.sigma.data)

    # use smaller dataset for parameter fitting
    fitting_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=8, shuffle=True, drop_last=True, num_workers=8
    )

    regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
        regularizer,
        physics,
        data_fidelity,
        lmbd,
        fitting_dataloader,
        val_dataloader,
        epochs=1,
        mode="JFB",
        NAG_step_size=1e-1,
        NAG_max_iter=1000,
        NAG_tol_train=1e-4,
        NAG_tol_val=1e-4,
        lr=0.01,
        lr_decay=0.95,
        device=device,
        verbose=False,
        dynamic_range_psnr=True,
        validation_epochs=1,
        logger=logger,
    )

    torch.save(regularizer.state_dict(), f"weights/LSR_pretraining_and_parameter_fitting_on_LoDoPaB.pt")

logger.info(f"Sigma {regularizer.sigma.data}, alpha: {regularizer.alpha.data}")
print(regularizer.sigma)
print(regularizer.alpha)

# bilevel training

for p in regularizer.parameters():
    p.requires_grad_(True)

regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=10,
    validation_epochs=1,
    mode="JFB",
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    lr=1e-5,
    lr_decay=0.9,
    device=device,
    verbose=False,
    dynamic_range_psnr=True,
    reg=False,
    reg_para=1e-5,
    reg_reduced=True,
    savestr="weights/LSR_jfb_CT",
    logger=logger,
)

torch.save(regularizer.state_dict(), f"weights/LSR_jfb_CT.pt")

# %%
