"""
Created on Mon May 26 2025

@author: Yasi Zhang
"""
from deepinv.physics import Denoising, MRI, GaussianNoise, Tomography
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from evaluation import evaluate
from dataset import get_dataset
from torchvision.transforms import CenterCrop, RandomCrop
from training_methods.simple_ar_training import estimate_lmbd
from priors import ICNNPrior, CNNPrior, linearICNNPrior, WCRR, linearIDCNNPrior
import torch
from operators import get_operator
if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "CT"

# problem dependent parameters
physics , data_fidelity = get_operator(problem, device)
lmbd = 1.0
noise_level = 0.1

train_dataset = get_dataset("LoDoPaB", test=False, transform=None)
# splitting in training and validation set
test_ratio = 0.1
val_len = int(len(train_dataset) * 0.1)
train_len = len(train_dataset) - val_len
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = torch.utils.data.Subset(train_dataset, range(train_len, len(train_dataset)))


# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=32, shuffle=True, drop_last=True, num_workers=8
)
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True, num_workers=8
)

 

regularizer = linearIDCNNPrior(in_channels=1,
    num_filters=32,
    kernel_dim=5,
    num_layers=5, pretrained="./weights/simple_linearIDCNNPrior_ar_CT.pt"
).to(device)

lmbd = estimate_lmbd(train_dataloader,physics,device)


 
NAG_step_size = 1e-1  # step size in NAG
NAG_max_iter = 500  # maximum number of iterations in NAG
NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)
only_first = False


### Evauate using NAG with backtracking
mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=val_dataloader.dataset,
    regularizer=regularizer,
    lmbd=lmbd,
    NAG_step_size=NAG_step_size,
    NAG_max_iter=NAG_max_iter,
    NAG_tol=NAG_tol,
    only_first=only_first,
    adaptive_range=True,
    device=device,
    verbose=True,
)

print('final psnr:')
print(mean_psnr)
# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
