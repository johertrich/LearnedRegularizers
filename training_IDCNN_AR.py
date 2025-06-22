#%%
from priors import simple_IDCNNPrior, linearIDCNNPrior
import torch
from deepinv.physics import Denoising, GaussianNoise
from deepinv.optim import L2
from dataset import get_dataset
from operators import get_operator
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
from training_methods import simple_ar_training
from training_methods.simple_ar_training import estimate_lmbd

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "Denoising"

  
 


physics , data_fidelity = get_operator(problem, device)

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
print('train_dataset length:',len(train_dataset))
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
regularizer = simple_IDCNNPrior(in_channels=1, channels=32, device=device, kernel_size=5)

lmbd = estimate_lmbd(train_dataloader,physics,device)
 

simple_ar_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    device=device,
    epochs=50,
    lr=1e-3,
    mu=10,
    save_str=f"./weights/simple_{regularizer.__class__.__name__}_ar_{problem}.pt"
)



