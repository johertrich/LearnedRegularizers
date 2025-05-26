from operators import get_evaluation_setting
from deepinv.utils.plotting import plot
from evaluation import evaluate
from dataset import get_dataset
from torchvision.transforms import CenterCrop
from priors import LSR
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

# Define regularizer

pretrained = "weights/LSR_jfb.pt"
regularizer = LSR(
    nc=[32, 64, 128, 256], pretrained_denoiser=False, pretrained=pretrained
).to(device)

# reconstruction hyperparameters, might be problem dependent
lmbd = 1.0

# Parameters for the Nesterov Algorithm, might also be problem dependent...

NAG_step_size = 1e-1  # step size in NAG
NAG_max_iter = 1000  # maximum number of iterations in NAG
NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)


#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)

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
    adaptive_range=False,
    device=device,
    verbose=True,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
