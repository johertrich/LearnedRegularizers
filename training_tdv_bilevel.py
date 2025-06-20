from priors import TDV
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
    device = "cuda:3"
else:
    device = "cpu"

problem = "Denoising"

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="training_TDV_bilevel_" + str(datetime.datetime.now()) + ".log",
    level=logging.INFO,
)

physics, data_fidelity = get_operator(problem, device)

transform = Compose(
    [
        RandomCrop(128),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomApply([RandomRotation((90, 90))], p=0.5),
    ]
)
train_dataset = get_dataset("BSDS500_gray", test=False, transform=transform)
val_dataset = get_dataset("BSDS500_gray", test=False, transform=CenterCrop(321))
# splitting in training and validation set
test_ratio = 0.1
test_len = int(len(train_dataset) * 0.1)
train_len = len(train_dataset) - test_len
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = get_dataset(
    "BSD68"
)  # torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=8, shuffle=True, drop_last=True, num_workers=8
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True, num_workers=8
)

# define regularizer
lmbd = 1.0
# define regularizer
config = dict(
    in_channels=1,
    num_features=32,
    multiplier=1,
    num_mb=3,
    num_scales=3,
    potential="lncosh",
    zero_mean=True,
    step_size_parameter=True,
)
regularizer = TDV(**config).to(device)

params = 0
for p in regularizer.parameters():
    params += p.numel()
print(params)

# Pretraining
load_pretrain = True
if load_pretrain:
    regularizer.load_state_dict(torch.load("weights/TDV_pretraining_on_BSD.pt"))
else:
    from tqdm import tqdm
    import numpy as np
    from deepinv.loss.metric import PSNR

    transform_pretrain = Compose(
        [
            RandomCrop(128),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomApply([RandomRotation((90, 90))], p=0.5),
        ]
    )
    pretrain_dataset = get_dataset(
        "BSDS500_gray", test=False, transform=transform_pretrain
    )
    pretrain_dataloader = torch.utils.data.DataLoader(
        pretrain_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=8
    )

    psnr = PSNR()
    epochs_pretraining = 5000
    noise_level_range = [5.0 / 255.0, 40.0 / 255.0]
    optimizer = torch.optim.Adam(regularizer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    for p in regularizer.parameters():
        p.requires_grad_(True)
    regularizer.step_size_scaling.requires_grad_(False)
    for epoch in range(epochs_pretraining):
        losses = []
        for x in tqdm(
            pretrain_dataloader, desc=f"Epoch {epoch+1}/{epochs_pretraining} - Train"
        ):
            optimizer.zero_grad()
            x = x.to(device)
            noisy = x + 0.03 * torch.randn_like(x)
            pred = noisy - regularizer.grad(noisy)
            loss = torch.sum(torch.abs(pred - x) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(regularizer.parameters(), 0.1)
            losses.append(loss.item())
            optimizer.step()
        print(np.mean(losses))
        if (epoch + 1) % 20 == 0:
            print("Validation")
            losses = []
            psnrs = []
            for x in tqdm(
                val_dataloader, desc=f"Epoch {epoch+1}/{epochs_pretraining} - Val"
            ):
                x = x.to(device)
                noisy = x + 0.03 * torch.randn_like(x)
                pred = noisy - regularizer.grad(noisy)
                loss = torch.sum(torch.abs(pred - x) ** 2)
                losses.append(loss.item())
                psnrs.append(psnr(pred, x).mean().item())
            print(np.mean(losses), np.mean(psnrs))
            logger.info(
                "Pretrain Validation Epoch {0} of {1}, loss {2:.2f}, PSNR {3:.2f}".format(
                    epoch + 1, epochs_pretraining, np.mean(losses), np.mean(psnrs)
                )
            )
            scheduler.step()

    torch.save(regularizer.state_dict(), f"weights/TDV_pretraining_on_BSD.pt")

# Parameter fitting

load_pretrain = True
if load_pretrain:
    regularizer.load_state_dict(torch.load("weights/TDV_fitted_on_BSD.pt"))
else:
    for p in regularizer.parameters():
        p.requires_grad_(False)
    regularizer.step_size_scaling.requires_grad_(True)


    regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
        regularizer,
        physics,
        data_fidelity,
        lmbd,
        train_dataloader,
        val_dataloader,
        epochs=20,
        mode="JFB",
        NAG_step_size=1e-1,
        NAG_max_iter=1000,
        NAG_tol_train=1e-4,
        NAG_tol_val=1e-4,
        lr=0.1,
        lr_decay=0.95,
        device=device,
        verbose=False,
        logger=logger,
    )

    torch.save(regularizer.state_dict(), f"weights/TDV_fitted_on_BSD.pt")

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
    epochs=200,
    mode="IFT",
    NAG_step_size=1e-1,
    NAG_max_iter=1500,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    lr=0.00005,
    lr_decay=0.99,
    reg=True,
    reg_para=5e-4,
    device=device,
    verbose=False,
    logger=logger,
    adabelief=True,
)

torch.save(regularizer.state_dict(), f"weights/TDV_bilevel_8.pt")

# %%
