from priors import ICNNPrior
import torch
from deepinv.physics import Denoising, GaussianNoise
from training_methods import simple_bilevel_training
from deepinv.optim import L2
from dataset import get_dataset
from torchvision.transforms import RandomCrop, CenterCrop, Resize
from torch.utils.data import Subset as subset

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "Denoising"
algorithm = "AdamW" # or "MAID", "Adam", "AdamW", "ISGD_Momentum"

# problem dependent parameters
if problem == "Denoising":
    noise_level = 0.1
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
    data_fidelity = L2(sigma=1.0)
    # Set the transform to a deterministic one like CenterCrop or Resize for MAID
    if algorithm == "MAID":
        crop = CenterCrop((64, 64))
    else:
        crop = RandomCrop((64, 64))
    dataset = get_dataset("BSDS500_gray", test=False, transform= crop)
    lmbd = 1.0


# splitting in training and validation set
test_ratio = 0.1
test_len = int(len(dataset) * 0.1)
train_len = len(dataset) - test_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])
batch_size = 8
shuffle = True
if algorithm == "MAID":
    training_size_full_batch = 32 # Can be increased up to GPU memory
    train_set = subset(train_set, list(range(training_size_full_batch)))
    batch_size = training_size_full_batch
    shuffle = False
# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=8, shuffle=True, drop_last=True
)

# define regularizer
regularizer = ICNNPrior(
    in_channels=1, strong_convexity= 0.0 , num_layers=3, num_filters=16
).to(device)

regularizer = simple_bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    device=device,
    verbose=False,
    optimizer_alg=algorithm,
    lr=1e-3, 
    NAG_tol = 1e-4, # Set 1e-1 for MAID
)
torch.save(regularizer.state_dict(), "weights/simple_ICNN_bilevel.pt")
