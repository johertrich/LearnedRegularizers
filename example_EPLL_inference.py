"""
Created on Wed Feb 26 2025

@author: Zakobian
"""
from deepinv.physics import Denoising, MRI, GaussianNoise, Tomography
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from evaluation import evaluate, nag
from dataset import get_dataset
from operators import MRIonR
from torchvision.transforms import CenterCrop, RandomCrop
import torch
from priors.newepll import EPLL

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
only_first = True  # just evaluate on the first image of the dataset for test purposes

############################################################

# reconstruction hyperparameters, might be problem dependent
if problem == "Denoising":
    lmbd = 1e-4  # regularization parameter
elif problem == "CT":
    lmbd = 500.0  # regularization parameter



#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator
if problem == "Denoising":
    noise_level = 0.1
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
    data_fidelity = L2(sigma=1.0)
    # dataset = get_dataset("BSD68")
    dataset = get_dataset("BSDS500_gray", test=True, transform=RandomCrop(300))
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

regularizer = EPLL(
    device=device, patch_size=8, channels=1, pretrained="gmm.pt"
)

# Parameters for the Nesterov Algorithm, might also be problem dependent...
NAG_step_size = 1e-2 #0.99/physics.compute_norm(dataset[0])  # step size in NAG
NAG_max_iter = 1000  # maximum number of iterations in NAG
NAG_tol = 1e-10  # tolerance for the relative error (stopping criterion)
beta = 0.9

# breakpoint()
### Evauate using NAG with backtracking
for lmbd in [1e-4]:
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
        verbose=False,
    )
    print(f"Mean PSNR over the test set: {mean_psnr:.2f} with lmbd={lmbd}")
    # print(f"Regularization parameter: {lmbd}")
breakpoint()