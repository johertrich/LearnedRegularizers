# Evaluation script for the columns BL+IFT, BL+JFB and ML for CRR, WCRR, ICNN, IDCNN, LAR, TDV and LSR

import torch
from priors import (
    ParameterLearningWrapper,
    WCRR,
    simple_ICNNPrior,
    simple_IDCNNPrior,
    LSR,
    TDV,
    LocalAR,
)
from deepinv.utils.plotting import plot
from evaluation import evaluate
from dataset import get_dataset
from operators import get_evaluation_setting

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

problem = "Denoising"  # Denoising or CT
evaluation_mode = "IFT"  # IFT, JFB or Score
regularizer_name = "CRR"  # CRR, WCRR, ICNN, IDCNN, LAR, TDV or LSR
only_first = False


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
        nc=[32, 64, 128, 256], pretrained_denoiser=False, alpha=1.0, sigma=score_sigma
    ).to(device)

regularizer = ParameterLearningWrapper(reg, device=device)
lmbd = 1.0

NAG_step_size = 1e-1  # step size in NAG
NAG_max_iter = 1000  # maximum number of iterations in NAG
NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)

if evaluation_mode == "Score":
    weights = torch.load(
        "weights/score_parameter_fitting_for_{problem}/{regularizer_name}_fitted_parameters_with_IFT_for_{problem}.pt",
        map_location=device,
    )
else:
    weights = torch.load(
        f"weights/bilevel_{problem}/{regularizer_name}_bilevel_{evaluation_mode}_for_{problem}.pt",
        map_location=device,
    )
regularizer.load_state_dict(weights)

dataset, physics, data_fidelity = get_evaluation_setting(problem, device)

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

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
