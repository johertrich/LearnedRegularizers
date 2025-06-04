from operators import get_evaluation_setting
from deepinv.optim import L2, Tikhonov
from deepinv.utils.plotting import plot
from evaluation import evaluate
from dataset import get_dataset
from torchvision.transforms import CenterCrop
from priors import ICNNPrior, wcrr
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

weakly = False
pretrained = "weights/WCRR_score.pt" if weakly else "weights/CRR_score.pt"
regularizer = wcrr.WCRR(
    sigma=0.1, weak_convexity=1.0 if weakly else 0.0, pretrained=pretrained
).to(device)

# reconstruction hyperparameters, might be problem dependent
if problem == "Denoising":
    lmbd = 2e-3  # regularization parameter
elif problem == "CT":
    lmbd = 10.0  # regularization parameter

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
    device=device,
    verbose=True,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
