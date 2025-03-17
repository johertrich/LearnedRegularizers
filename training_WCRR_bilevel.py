#%%
from priors import WCRR
import torch
from deepinv.physics import Denoising, GaussianNoise
from training_methods import simple_bilevel_training_adam, simple_bilevel_training_maid
from deepinv.optim import L2
from dataset import get_dataset
from torchvision.transforms import RandomCrop, CenterCrop, Resize, Compose
from torch.utils.data import Subset as subset

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "Denoising"
algorithm = "Adam"  #or "MAID"

# problem dependent parameters
if problem == "Denoising":
    noise_level = 0.1
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
    data_fidelity = L2(sigma=1.0)
    # Set the transform to a deterministic one like CenterCrop or Resize for MAID
    if algorithm == "MAID":
        crop_train = CenterCrop((64, 64))
    else:
        crop_train = RandomCrop((64, 64))
    crop_val = CenterCrop((64, 64))
    dataset = get_dataset("BSDS500_gray", test=False, transform=None)
    lmbd = 1.0

# splitting in training and validation set
test_ratio = 0.1
test_len = int(len(dataset) * 0.1)
train_len = len(dataset) - test_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])
train_trans = train_set.dataset.transforms
val_trans = val_set.dataset.transforms
train_set.dataset.transforms = Compose([train_trans,crop_train])
val_set.dataset.transforms = Compose([val_trans,crop_val])
batch_size = 48
shuffle = True
if algorithm == "MAID":
    training_size_full_batch = 32  # Can be increased up to GPU memory
    train_set = subset(train_set, list(range(training_size_full_batch)))
    batch_size = training_size_full_batch
    shuffle = False
# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=8, shuffle=False, drop_last=True
)

# define regularizer
regularizer = WCRR(
    sigma=0.1, weak_convexity=1.0,
).to(device)

if algorithm == 'Adam':    
    regularizer, loss_train, loss_val, psnr_train, psnr_val = simple_bilevel_training_adam(
        regularizer,
        physics,
        data_fidelity,
        lmbd,
        train_dataloader,
        val_dataloader,
        epochs=100,
        NAG_step_size=1e-3,
        NAG_max_iter=500,
        NAG_tol_train=1e-4,
        NAG_tol_val=1e-6,
        linesearch=True,
        lr = 0.005,
        lr_decay=0.9,
        device=device,
        verbose=False,
        ICNN=False
    )
else:
    regularizer, loss_train, loss_val, psnr_train, psnr_val = simple_bilevel_training_maid(
        regularizer,
        physics,
        data_fidelity,
        lmbd,
        train_dataloader,
        val_dataloader,
        epochs=100,
        NAG_step_size=1e-3,
        NAG_max_iter=500,
        NAG_tol_train=1e-4,
        NAG_tol_val=1e-6,
        linesearch=True,
        lr = 0.005,
        lr_decay=0.9,
        device=device,
        verbose=False,
        ICNN=False
    )
torch.save(regularizer.state_dict(), f'weights/WCRR_bilevel_{algorithm}.pt')

# %%
