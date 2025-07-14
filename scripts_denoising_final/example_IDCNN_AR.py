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
from operators import get_evaluation_setting
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from evaluation import evaluate

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "Denoising"

  


# define regularizer
regularizer = simple_IDCNNPrior(in_channels=1, channels=32, device=device, kernel_size=5,
    pretrained='./weights/simple_simple_IDCNNPrior_ar_Denoising.pt'
).to(device)

 
 

# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)
lmbd = 5

# Call unified evaluation routine

NAG_step_size = 1e-2  # step size in NAG
NAG_max_iter = 1000  # maximum number of iterations in NAG
NAG_tol = 1e-4  # tolerance for therelative error (stopping criterion)
only_first = False

mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
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

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
