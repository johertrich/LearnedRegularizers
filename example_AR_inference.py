"""
Created on Wed Feb 26 2025

@author: Zakobian
"""
from deepinv.physics import Denoising, MRI, GaussianNoise, Tomography
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from evaluation import evaluate
from dataset import get_dataset
from torchvision.transforms import CenterCrop, RandomCrop
from training_methods.simple_ar_training import estimate_lmbd
from priors import ICNNPrior, CNNPrior, linearICNNPrior, WCRR
import torch

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device: ", device)
torch.random.manual_seed(0)  # make results deterministic

############################################################

# Problem selection

problem = "Denoising"  # Select problem setups, which we consider.
only_first = False  # just evaluate on the first image of the dataset for test purposes

############################################################

# reconstruction hyperparameters, might be problem dependent
if problem == "Denoising":
    lmbd = 1.0#20.0  # regularization parameter
elif problem == "CT":
    lmbd = 500.0  # regularization parameter

# Parameters for the Nesterov Algorithm, might also be problem dependent...

NAG_step_size = 1e-1  # step size in NAG
NAG_max_iter = 500  # maximum number of iterations in NAG
NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)
beta = 0.9

#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator
if problem == "Denoising":
    noise_level = 0.1
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
    data_fidelity = L2(sigma=1.0)
    #dataset = get_dataset("BSD68")
    # dataset = get_dataset("BSDS500_gray", test=True, transform=RandomCrop(64))
    # dataset = get_dataset("BSDS500_gray", test=True, transform=CenterCrop(300))
    dataset = get_dataset("BSD68", transform=CenterCrop(300))
elif problem == "CT":
    noise_level = 0.5
    dataset = get_dataset("BSD68", transform=CenterCrop(300))
    imsize = dataset[0].shape[-1]
    physics = Tomography(
        imsize // 3, imsize, device=device, noise_model=GaussianNoise(sigma=noise_level)
    )
    data_fidelity = L2(sigma=1.0)

else:
    raise NotImplementedError("Problem not found")

# Call unified evaluation routine
# Define regularizer

# regularizer = ICNNPrior(
#     in_channels=1,
#     strong_convexity=0,
#     num_layers=5,
#     num_filters=16,
#     pretrained="./weights/simple_ICNNPrior_ar.pt",
#     # pretrained="./weights/simple_ICNN_unrolling.pt",
#     device=device,
# )
# regularizer = linearICNNPrior(
#     in_channels=1,
#     strong_convexity=0,
#     num_layers=5,
#     num_filters=16,
#     pretrained="./weights/simple_linearICNNPrior_ar.pt",
#     # pretrained="./weights/simple_ICNN_unrolling.pt",
#     device=device,
# )
weakly=True
regularizer = WCRR(
    sigma=0.1, weak_convexity=1.0 if weakly else 0.0, pretrained="./weights/simple_WCRR_ar.pt"
).to(device)


### Estimatee the reg parameter.
lmbd = estimate_lmbd(dataset, physics, device)


### Evauate using NAG with backtracking
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
    device=device,
    verbose=True,
)

### Evauate using NAG
# mean_psnr, x_out, y_out, recon_out = evaluate(
#     physics=physics,
#     data_fidelity=data_fidelity,
#     dataset=dataset,
#     regularizer=regularizer,
#     lmbd=lmbd,
#     NAG_step_size=NAG_step_size,
#     NAG_max_iter=NAG_max_iter,
#     NAG_tol=NAG_tol,
#     only_first=only_first,
#     device=device,
#     verbose=True,
# )

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])

