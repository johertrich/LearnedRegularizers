# Evaluation script for the columns BL+IFT, BL+JFB, ML and adversarial for CRR, WCRR, ICNN, IDCNN, TDV, LAR, LSR and NETT. If regularizer_name == "NETT", No need to specify the other parse arguments.

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import argparse
from dataset import get_dataset
from priors import (
    NETT,
    LSR,
    WCRR,
    ICNNPrior,
    IDCNNPrior,
    TDV,
    ParameterLearningWrapper,
    LocalAR,
    EPLL,
    PatchNR,
)
from deepinv.optim.utils import GaussianMixtureModel
from operators import get_operator, get_evaluation_setting
from evaluation import evaluate
from deepinv.utils.plotting import plot
import logging
import datetime

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

torch.random.manual_seed(1)  # make results deterministic

parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument("--evaluation_mode", type=str, default="IFT")
parser.add_argument("--problem", type=str, default="Denoising")
parser.add_argument("--regularizer_name", type=str, default="CRR")
parser.add_argument("--only_first", type=bool, default=False)
parser.add_argument("--save_results", type=bool, default=False)
inp = parser.parse_args()

problem = inp.problem  # Denoising or CT
evaluation_mode = inp.evaluation_mode  # AR, IFT, JFB, NETT or Score
regularizer_name = (
    inp.regularizer_name
)  # CRR, WCRR, ICNN, IDCNN, TDV, LAR, LSR and NETT
only_first = inp.only_first
save_results = inp.save_results  # If True, save the first 10 image reconstructions

if save_results:
    save_path = f"savings/{problem}/{regularizer_name}/{evaluation_mode}"
    space = evaluation_mode
    if regularizer_name == "NETT":
        save_path = f"savings/{problem}/{regularizer_name}"
        space = ""
    logging_path = save_path + "/logging"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(logging_path, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logging_path
        + "/log_eval_"
        + problem
        + "_"
        + regularizer_name
        + "_"
        + space
        + "_"
        + str(datetime.datetime.now())
        + ".log",
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
    )
    logger.info(f"Evaluation {regularizer_name} with {evaluation_mode} on {problem}!!!")

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
    reg = ICNNPrior(in_channels=1, channels=32, device=device, kernel_size=5).to(device)
elif regularizer_name == "IDCNN":
    reg = IDCNNPrior(in_channels=1, channels=32, device=device, kernel_size=5).to(
        device
    )
elif regularizer_name == "LAR":
    reg = LocalAR(
        in_channels=1,
        pad=True,
        use_bias=False,
        n_patches=-1,
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
elif regularizer_name == "NETT":
    reg = NETT(
        in_channels=1, out_channels=1, hidden_channels=64, padding_mode="zeros"
    ).to(device)
elif regularizer_name == "EPLL":
    weights_filepath = f"weights/gmm_{problem}.pt"
    setup_data = torch.load(weights_filepath)
    patch_size = setup_data["patch_size"]
    n_gmm_components = setup_data["n_gmm_components"]
    GMM = GaussianMixtureModel(n_gmm_components, patch_size**2, device=device)
    GMM.load_state_dict(setup_data["weights"])
    regularizer = EPLL(
        device=device,
        patch_size=patch_size,
        channels=1,
        n_gmm_components=n_gmm_components,
        GMM=GMM,
        pad=True,
        batch_size=30000,
    )
    lmbd = setup_data["lambda"]
elif regularizer_name == "PatchNR":
    train_on = "BSD500" if problem == "Denoising" else "LoDoPaB"
    regularizer = PatchNR(
        patch_size=6,
        channels=1,
        num_layers=5,
        sub_net_size=512,
        device=device,
        n_patches=-1,
        pretrained=f"weights/patchnr_6x6_{train_on}.pt",
        pad=False,
    )
    lmbd = 21.0
else:
    raise ValueError("Unknown model!")


if evaluation_mode == "AR":
    weights = torch.load(
        f"weights/adversarial_{problem}/{regularizer_name}_adversarial_for_{problem}_fitted.pt",
        map_location=device,
        weights_only=True,
    )
elif evaluation_mode == "Score":
    if not regularizer_name in ["EPLL", "PatchNR", "LAR"]:
        weights = torch.load(
            f"weights/score_parameter_fitting_for_{problem}/{regularizer_name}_fitted_parameters_with_IFT_for_{problem}.pt",
            map_location=device,
            weights_only=True,
        )
    if regularizer_name == "LAR":
        weights = torch.load(
            f"weights/score_parameter_fitting_for_{problem}/{regularizer_name}_fitted_parameters_with_JFB_for_{problem}.pt",
            map_location=device,
            weights_only=True,
        )
elif evaluation_mode == "NETT":
    weights = torch.load(
        f"weights/NETT_{problem}_fitted.pt",
        map_location=device,
        weights_only=True,
    )
else:
    weights = torch.load(
        f"weights/bilevel_{problem}/{regularizer_name}_bilevel_{evaluation_mode}_for_{problem}.pt",
        map_location=device,
        weights_only=True,
    )

if not regularizer_name in ["EPLL", "PatchNR"]:
    regularizer = ParameterLearningWrapper(reg, device=device)
    regularizer.load_state_dict(weights)
    lmbd = 1.0

# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)
NAG_step_size = (
    1e-3 if regularizer_name in ["EPLL", "PatchNR"] else 1e-1
)  # step size in NAG
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
    adaptive_range=problem == "CT",
    device=device,
    adam=regularizer_name in ["EPLL", "PatchNR"],
    verbose=True,
    save_path=save_path if save_results else None,
    logger=logger if save_results else None,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
