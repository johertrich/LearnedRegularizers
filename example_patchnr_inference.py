"""

@author: Alex
"""

from deepinv.physics import Denoising, MRI, GaussianNoise, Tomography
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from evaluation import evaluate
from dataset import get_dataset
from torchvision.transforms import CenterCrop, RandomCrop
import torch
import time

from priors.patchnr import PatchNR


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

problem = "CT"  # Select problem setups, which we consider.
only_first = True  # just evaluate on the first image of the dataset for test purposes

############################################################

# reconstruction hyperparameters, might be problem dependent
if problem == "Denoising":
    lmbd = 21.0  # regularization parameter
elif problem == "CT":
    lmbd = 300.0  # regularization parameter


#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator
if problem == "Denoising":
    noise_level = 0.1
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
    data_fidelity = L2(sigma=1.0)
    dataset = get_dataset("BSD68")
    # dataset = get_dataset("BSDS500_gray", test=True, transform=RandomCrop(300))
    dataset = torch.utils.data.Subset(dataset, range(2, 3))
elif problem == "CT":
    noise_level = 0.5
    dataset = get_dataset("BSDS500_gray", test=True, transform=CenterCrop(300))
    dataset = torch.utils.data.Subset(dataset, range(0, 10))
    imsize = dataset[0].shape[-1]
    physics = Tomography(
        imsize // 3, imsize, device=device, noise_model=GaussianNoise(sigma=noise_level)
    )
    data_fidelity = L2(sigma=1.0)

else:
    raise NotImplementedError("Problem not found")

# Call unified evaluation routine
# Define regularizer

regularizer = PatchNR(
    patch_size=6,
    channels=1,
    num_layers=5,
    sub_net_size=512,
    device=device,
    n_patches=-1,
    pretrained="weights/patchnr.pt",
    pad=False,
)

NAG_step_size = 1e-1  # step size in NAG
NAG_max_iter = 500  # maximum number of iterations in NAG
NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)

start = time.time()
### Evauate using NAG
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
end = time.time()
print(f"TIME: {end-start}s")
# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
