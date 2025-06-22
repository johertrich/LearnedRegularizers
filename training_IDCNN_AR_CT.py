#%%
from priors import simple_IDCNNPrior, linearIDCNNPrior, simple_IDCNNPrior_final
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

problem = "CT"

  
 


physics , data_fidelity = get_operator(problem, device)

train_dataset = get_dataset("LoDoPaB", root='/home/yasmin/projects/backup/LearnedRegularizers/', test=False, transform=None)
# splitting in training and validation set
test_ratio = 0.003
val_len = int(len(train_dataset) * test_ratio)
train_len = len(train_dataset) - val_len
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = torch.utils.data.Subset(train_dataset, range(train_len, len(train_dataset)))
print(len(train_set), len(val_set))

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=32, shuffle=True, drop_last=True, num_workers=8
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True, num_workers=8
)

# define regularizer
regularizer = simple_IDCNNPrior(in_channels=1, channels=32, device=device, kernel_size=5, )



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


# torch.save(regularizer.state_dict(), f"weights/{idx}/simple_{regularizer.__class__.__name__}_ar_{problem}.pt")
# %%
