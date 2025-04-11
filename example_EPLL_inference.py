from deepinv.physics import Denoising, MRI, GaussianNoise, Tomography
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from evaluation import evaluate, nag
from dataset import get_dataset
from operators import MRIonR
from torchvision.transforms import CenterCrop, RandomCrop
import torch
from priors.epll import EPLL

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
    lmbd = 1e1  # regularization parameter
elif problem == "CT":
    lmbd = 275  # 3e2  # regularization parameter


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
    dataset = get_dataset("BSDS500_gray", test=True, transform=CenterCrop(300))
    imsize = dataset[0].shape[-1]
    physics = Tomography(
        imsize // 3, imsize, device=device, noise_model=GaussianNoise(sigma=noise_level)
    )
    data_fidelity = L2(sigma=1.0)

else:
    raise NotImplementedError("Problem not found")

regularizer = EPLL(
    device=device,
    patch_size=8,
    channels=1,
    n_gmm_components=200,
    pretrained="weights/gmm_8x8patchsize_200components.pt",
)

# Parameters for the Nesterov Algorithm, might also be problem dependent...
NAG_step_size = 1e-4  # step size for NAG
NAG_max_iter = 500  # 1000  # maximum number of NAG iterations
NAG_tol = 1e-6  # tolerance for the relative error (stopping criterion)
beta = 0.9

### Evauate using NAG with backtracking
# for lmbd in [1e1, 3e1, 5e1, 7e1, 1e2]:
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
print(f"Mean PSNR over the test set: {mean_psnr:.2f} with lmbd={lmbd}")
plot([x_out, y_out, recon_out])
# breakpoint()
