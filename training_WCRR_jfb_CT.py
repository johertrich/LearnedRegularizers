from priors import WCRR
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

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "CT"

physics, data_fidelity = get_operator(problem, device)

train_dataset = get_dataset("LoDoPaB", test=False)
val_dataset = get_dataset("LoDoPaB", test=False)
# splitting in training and validation set
test_ratio = 0.1
test_len = int(len(train_dataset) * 0.1)
train_len = len(train_dataset) - test_len
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))


# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=8, shuffle=True, drop_last=True, num_workers=8
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True, num_workers=8
)

# define regularizer
noise_level = 0.1
lmbd = 1.0
regularizer = WCRR(
    sigma=noise_level,
    weak_convexity=1.0,
).to(device)

regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=2,
    mode="JFB",
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    lr=0.005,
    lr_decay=0.5,
    validation_epochs=1,
    device=device,
    verbose=False,
)

torch.save(regularizer.state_dict(), f"weights/WCRR_jfb_CT.pt")

# %%
