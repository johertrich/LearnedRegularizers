# Evaluation script for the columns BL+IFT, BL+JFB, ML and adversarial for CRR, WCRR, ICNN, IDCNN, TDV, LAR and LSR

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import argparse
from dataset import get_dataset
from priors import (
	LSR, 
	WCRR, 
	simple_ICNNPrior, 
	simple_IDCNNPrior, 
	TDV, 
	ParameterLearningWrapper,
	LocalAR,
)
from operators import get_operator, get_evaluation_setting
from evaluation import evaluate
from deepinv.utils.plotting import plot

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

torch.random.manual_seed(0)  # make results deterministic

parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument("--evaluation_mode", type=str, default="IFT")
parser.add_argument("--problem", type=str, default="Denoising")
parser.add_argument("--regularizer_name", type=str, default="CRR")
parser.add_argument("--only_first", type=bool, default=False)
inp=parser.parse_args()

problem = inp.problem  # Denoising or CT
evaluation_mode = inp.evaluation_mode  # AR, IFT, JFB or Score
regularizer_name = inp.regularizer_name # CRR, WCRR, ICNN, IDCNN, TDV, LAR or LSR
only_first = inp.only_first

# define regularizer
if regularizer_name == "CRR":
    reg = WCRR(
        sigma=0.1,
        weak_convexity=0.0,
    ).to(device)
elif regularizer_name == "WCRR":
    reg = WCRR(
        sigma=0.1,
        weak_convexity=1.0,
    ).to(device)
elif regularizer_name == "ICNN":
    reg = simple_ICNNPrior(in_channels=1, channels=32, device=device, kernel_size=5).to(
        device
    )
elif regularizer_name == "IDCNN":
    reg = simple_IDCNNPrior(
        in_channels=1, channels=32, device=device, kernel_size=5
    ).to(device)
elif regularizer_name == "LAR":
    reg = LocalAR(
        in_channels=1,
        pad=True,
        use_bias=False,
        n_patches=-1,
        normalise_grad=False,
        reduction="sum",
        output_factor=1 / 142**2,
        pretrained=None,
    ).to(device)
elif regularizer_name == "TDV":
    config = dict(
        in_channels=1,
        num_features=32,
        multiplier=1,
        num_mb=3,
        num_scales=3,
        potential="quadratic",
        activation="softplus",
        zero_mean=True,
    )
    reg = TDV(**config).to(device)
elif regularizer_name == "LSR":
    reg = LSR(
        nc=[32, 64, 128, 256], pretrained_denoiser=False, alpha=1.0, sigma=3e-2
    ).to(device)
else:
    raise ValueError("Unknown model!")

regularizer = ParameterLearningWrapper(reg, device=device)
lmbd = 1.0

if evaluation_mode == "AR":
    weights = torch.load(
    	f"weights/adversarial_{problem}/{regularizer_name}_adversarial_for_{problem}_fitted.pt",
        map_location=device,
        weights_only=True,
    )
elif evaluation_mode == "Score":
    weights = torch.load(
        f"weights/score_parameter_fitting_for_{problem}/{regularizer_name}_fitted_parameters_with_IFT_for_{problem}.pt",
        map_location=device,
        weights_only=True,
    )
else:
    weights = torch.load(
        f"weights/bilevel_{problem}/{regularizer_name}_bilevel_{evaluation_mode}_for_{problem}.pt",
        map_location=device,
        weights_only=True,
    )

regularizer.load_state_dict(weights)

# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)
test_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False
    )
NAG_step_size = 1e-1 # step size in NAG    
NAG_max_iter = 1000  # maximum number of iterations in NAG
NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)

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
    adaptive_range=True,
    device=device,
    verbose=True,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
