

from priors import NETT
import torch
from deepinv.physics import Denoising, GaussianNoise, Tomography
from training_methods import NETT_training
from deepinv.optim import L2, L1
from dataset import get_dataset
from operators import get_operator
from dataset.utils import NETT_transform
from torchvision.transforms import RandomCrop, RandomAutocontrast,CenterCrop, Resize, RandomHorizontalFlip, RandomVerticalFlip,RandomRotation
from torchvision.transforms.v2 import GaussianNoise, RandomApply
from torch.utils.data import Subset as subset
from torchvision import transforms

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "Tomography"
algorithm = "Adam"  # or "MAID", "Adam", "AdamW", "ISGD_Momentum"

# problem dependent parameters
if problem == "Tomography":
    dataset = get_dataset("LoDoPaB", test=False)
    physics, data_fidelity = get_operator("CT", device = device)
    # Set the transform to a deterministic one like CenterCrop or Resize for MAID
    NT = NETT_transform(0.5,physics)
    NT_val = NETT_transform(1,physics)
    NT_val.val = True
    #resize_trans = Resize((128,128))
    tran = transforms.Compose([RandomHorizontalFlip(),RandomVerticalFlip(),NT])
    tran_val = transforms.Compose([NT_val])
    dataset = get_dataset("LoDoPaB", test=False, transform = tran)
    dataset_val = get_dataset("LoDoPaB", test=False, transform = tran_val)
    lmbd = 1.0


if problem == "Denoising":
    noise_level = 0.1
    physics, data_fidelity = get_operator('Denoising',device = device)
    physics.device = device
    data_fidelity = L2(sigma=1.0)
    NT = NETT_transform(0.5,physics)
    NT_val = NETT_transform(1,physics)
    NT_val.val = True
    tran = transforms.Compose([RandomHorizontalFlip(),RandomVerticalFlip(),NT])
    tran_val = transforms.Compose([RandomHorizontalFlip(),RandomVerticalFlip(),NT_val])
    dataset = get_dataset("BSDS500_gray", test=False, transform=tran)
    dataset_val = get_dataset("BSDS500_gray", test=False, transform=tran_val)
    lmbd = 1



# splitting in training and validation set
test_ratio = 0.1
test_len = int(len(dataset) * 0.1)
train_len = len(dataset) - test_len

indices = torch.randperm(len(dataset))
train_indices = indices[:train_len]
val_indices = indices[test_len:]
train_set = torch.utils.data.Subset(dataset, train_indices)
val_set = torch.utils.data.Subset(dataset, val_indices)
batch_size = 4
shuffle = True

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, drop_last=True
)

# define regularizer
regularizer = NETT(in_channels = 1, out_channels = 1).to(device)#DRUNet(in_channels=1,out_channels = 1,device = device)
#regularizer.compute_padding((dataset[torch.tensor(0)][0].shape[-2],dataset[torch.tensor(0)][0].shape[-1]))

regularizer = NETT_training(
    regularizer,
    train_dataloader,
    val_dataloader,
    device=device,
    optimizer=algorithm,
    lr=1e-4,#1e-4
    num_epochs = 3000,
    save_best = True,
    weight_dir = "weights/NETT"+problem
)





