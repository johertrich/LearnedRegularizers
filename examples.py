from deepinv.physics import Denoising, MRI, GaussianNoise
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from evaluation import evaluate
from datasets import get_dataset
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"  # select device
torch.random.manual_seed(0)  # make results deterministic

############################################################

# Problem selection

problem = "Denoising"  # Select problem setups, which we consider.
only_first = True  # just evaluate on the first image of the dataset for test purposes

############################################################

# Define regularizer

regularizer = Tikhonov()
lmbd = 1.0  # regularization parameter

# Parameters for the Nesterov Algorithm

NAG_step_size = 1e-2  # step size in NAG
NAG_max_iter = 500  # maximum number of iterations in NAG
NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)


#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator

if problem == "Denoising":
    noise_level = 0.1
    physics = Denoising(GaussianNoise(sigma=noise_level))
    data_fidelity = L2(sigma=noise_level)
elif problem == "MRI":
    raise NotImplementedError("MRI not implemented")
else:
    raise NotImplementedError("Problem not found")

# Test dataset

dataset = get_dataset("BSDS500_gray")

# Call unified evaluation routine

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
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot(
    [x_out, y_out, recon_out], titles=["ground truth", "observation", "reconstruction"]
)
