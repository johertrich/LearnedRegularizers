from priors import ICNNPrior
import torch
from deepinv.physics import Denoising, GaussianNoise
from training_methods import simple_unrolling_training
from deepinv.optim import L2
from dataset import get_dataset
from torchvision.transforms import RandomCrop

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
problem = "Denoising"

# problem dependent parameters
if problem == "Denoising":
    noise_level = 0.1
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
    data_fidelity = L2(sigma=1.0)
    dataset = get_dataset("BSDS500_gray", test=False, transform=RandomCrop(64))
    lmbd = 1


# splitting in training and validation set
test_ratio = 0.1
test_len = int(len(dataset) * 0.1)
train_len = len(dataset) - test_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=8, shuffle=True, drop_last=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=8, shuffle=False, drop_last=True
)

# define regularizer
regularizer = ICNNPrior(in_channels=1, strong_convexity=0, num_layers=3, num_filters=16).to(device)


simple_unrolling_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    device = device
)

torch.save(regularizer.state_dict(), "weights/simple_ICNN_unrolling.pt")
