#%%
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from priors import WCRR, ParameterLearningWrapper
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
from training_methods.simple_ar_training import estimate_lmbd, estimate_lip
from operators import get_evaluation_setting
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
regularizer = WCRR(
    sigma=0.1,
    weak_convexity=0.0,
    ).to(device)
ckp = torch.load(f"weights/adversarial_{problem}/CRR_adversarial_for_{problem}.pt")
regularizer.load_state_dict(ckp)

# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)
lmbd = estimate_lmbd(dataset,physics,device)
lip = estimate_lip(regularizer,dataset,device)
lmbd_est = lmbd/lip

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
    lmbd=lmbd_est,
    NAG_step_size=NAG_step_size,
    NAG_max_iter=NAG_max_iter,
    NAG_tol=NAG_tol,
    only_first=only_first,
    adaptive_range=False,
    device=device,
    verbose=True,
)

wrapped_regularizer = ParameterLearningWrapper(regularizer, device=device)
ckp = torch.load(f"weights/adversarial_{problem}/CRR_adversarial_for_{problem}_fitted.pt")
wrapped_regularizer.load_state_dict(ckp)

mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
    regularizer=wrapped_regularizer,
    lmbd=1,
    NAG_step_size=NAG_step_size,
    NAG_max_iter=NAG_max_iter,
    NAG_tol=NAG_tol,
    only_first=only_first,
    adaptive_range=False,
    device=device,
    verbose=True,
    )

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
