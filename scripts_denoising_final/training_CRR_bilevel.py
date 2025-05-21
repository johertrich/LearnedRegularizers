from priors import wcrr
import torch
from deepinv.physics import Denoising, GaussianNoise
from training_methods import bilevel_training
from deepinv.optim import L2
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

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "Denoising"

physics, data_fidelity = get_operator(problem, device)

transform = Compose(
    [
        RandomCrop(64),
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
val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))



# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=32, shuffle=True, drop_last=True, num_workers=8
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True, num_workers=8
)

# define regularizer
noise_level = 0.1
lmbd = 1.0
regularizer = wcrr.WCRR(
    sigma=noise_level,
    weak_convexity=0.0,
).to(device)

regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=200,
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    lr=0.01,
    lr_decay=0.99,
    device=device,
    verbose=False,
)

torch.save(regularizer.state_dict(), f"weights/CRR_bilevel.pt")

# %%
