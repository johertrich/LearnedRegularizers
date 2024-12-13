from deepinv.physics import Denoising, MRI, GaussianNoise
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from evaluation import evaluate
from dataset import get_dataset
from operators import MRIonR
from torchvision.transforms import CenterCrop
from priors import ICNNPrior
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"  # select device
torch.random.manual_seed(0)  # make results deterministic

############################################################

# Problem selection

problem = "MRI"  # Select problem setups, which we consider.
only_first = True  # just evaluate on the first image of the dataset for test purposes

############################################################

# Define regularizer

regularizer = ICNNPrior(
    in_channels=1,
    strong_convexity=0,
    num_layers=3,
    num_filters=16,
    pretrained="weights/simple_ICNN_unrolling.pt",
)

# reconstruction hyperparameters, might be problem dependent
if problem == "Denoising":
    lmbd = 20.0  # regularization parameter
elif problem == "MRI":
    lmbd = 0.1  # regularization parameter

# Parameters for the Nesterov Algorithm, might also be problem dependent...

NAG_step_size = 1e-4  # step size in NAG
NAG_max_iter = 500  # maximum number of iterations in NAG
NAG_tol = 1e-6  # tolerance for the relative error (stopping criterion)


#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator

if problem == "Denoising":
    noise_level = 0.1
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
    data_fidelity = L2(sigma=1.0)
    dataset = get_dataset("BSDS500_gray", test=True)
elif problem == "MRI":
    dataset = get_dataset("BSDS500_gray", transform=CenterCrop(256), test=True)
    img_size = dataset[0].shape
    noise_level = 0.05
    # simple Cartesian mask generation from the deepinv tour...
    mask = torch.rand((1, img_size[-1]), device=device) > 0.75
    mask = torch.ones((img_size[-2], 1), device=device) * mask
    mask[:, int(img_size[-1] / 2) - 2 : int(img_size[-1] / 2) + 2] = 1
    # The MRI operator in deepinv operates on complex-valued images.
    # The MRIonR operator wraps it for real-valued images
    physics = MRIonR(
        mask=mask, device=device, noise_model=GaussianNoise(sigma=noise_level)
    )
    data_fidelity = L2(sigma=1.0)
else:
    raise NotImplementedError("Problem not found")

# Test dataset


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
