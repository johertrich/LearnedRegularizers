from priors import (
    ParameterLearningWrapper,
    WCRR,
    simple_ICNNPrior,
    simple_IDCNNPrior,
    LSR,
    TDV,
    LocalAR,
)
from training_methods import bilevel_training
import torch
from operators import get_evaluation_setting
from dataset import get_dataset
from evaluation import evaluate
import argparse
import os

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument("--evaluation_mode", type=str, default="bilevel-IFT")
parser.add_argument("--regularizer_name", type=str, default="CRR")
parser.add_argument("--load_fitted_parameters", type=bool, default=False)
inp = parser.parse_args()

problem = "CT"
regularizer_name = inp.regularizer_name
evaluation_mode = inp.evaluation_mode
load_fitted_parameters = inp.load_fitted_parameters
mode = "IFT"

lmbd_initial_guess = 60
lr=0.1

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

if (
    evaluation_mode == "bilevel-IFT"
    or evaluation_mode == "bilevel-JFB"
    or evaluation_mode == "Score"
):
    if regularizer_name == "IDCNN" or "LAR":
        mode = "JFB"
    if regularizer_name == "LSR":
        lr=0.01
    regularizer = ParameterLearningWrapper(reg, device=device)
    if evaluation_mode == "Score":
        weights = torch.load(
            f"weights/score_parameter_fitting_for_Denoising/{regularizer_name}_fitted_parameters_with_IFT_for_Denoising.pt",
            map_location=device,
        )
    elif evaluation_mode == "bilevel-IFT":
        weights = torch.load(
            f"weights/bilevel_Denoising/{regularizer_name}_bilevel_IFT_for_Denoising.pt",
            map_location=device,
        )
    elif evaluation_mode == "bilevel-JFB":
        weights = torch.load(
            f"weights/bilevel_Denoising/{regularizer_name}_bilevel_JFB_for_Denoising.pt",
            map_location=device,
        )
    regularizer.load_state_dict(weights)
elif evaluation_mode == "AR":
    if regularizer_name == "IDCNN":
        pretrained = "./weights/simple_simple_IDCNNPrior_ar_Denoising.pt"
        regularizer.load_state_dict(torch.load(pretrained, map_location=device))
    elif regularizer_name == "ICNN":
        pretrained = "./weights/simple_simple_ICNNPrior_ar_Denoising.pt"
        cpk = torch.load(pretrained, map_location=device)
        regularizer.load_state_dict(ckp)
    else:
        raise ValueError(f"no configuration for AR with regularizer {regularizer_name}")
else:
    raise ValueError("Unknown evaluation mode!")


if not os.path.isdir("weights"):
    os.mkdir("weights")
if not os.path.isdir(f"weights/Denoising_to_CT"):
    os.mkdir(f"weights/Denoising_to_CT")

dataset, physics, data_fidelity = get_evaluation_setting(problem, device)

validation_dataset = get_dataset("LoDoPaB_val")
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset, batch_size=5, shuffle=False, drop_last=False, num_workers=8
)

if isinstance(regularizer, ParameterLearningWrapper):
    wrapped_regularizer = regularizer
else:
    wrapped_regularizer = ParameterLearningWrapper(regularizer, device=device)

if load_fitted_parameters:
    wrapped_regularizer.load_state_dict(
        torch.load(
            f"weights/Denoising_to_CT/{regularizer_name}_with_{evaluation_mode}",
            map_location=device,
        )
    )
elif False:
    for p in wrapped_regularizer.parameters():
        p.requires_grad_(False)
    wrapped_regularizer.alpha.requires_grad_(True)
    wrapped_regularizer.scale.requires_grad_(True)

    # parameter search
    wrapped_regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
        wrapped_regularizer,
        physics,
        data_fidelity,
        lmbd_initial_guess,
        validation_dataloader,
        validation_dataloader,
        epochs=100,
        mode=mode,
        NAG_step_size=1e-1,
        NAG_max_iter=1000,
        NAG_tol_train=1e-4,
        NAG_tol_val=1e-4,
        lr=lr,
        lr_decay=0.999,
        device=device,
        verbose=False,
        validation_epochs=100,
        dynamic_range_psnr=True,
    )
    torch.save(
        wrapped_regularizer.state_dict(),
        f"weights/Denoising_to_CT/{regularizer_name}_with_{evaluation_mode}",
    )

print("Final alpha: ", wrapped_regularizer.alpha)
print("Final scale: ", wrapped_regularizer.scale)

only_first = False
wrapped_regularizer.alpha.requires_grad_(False)
wrapped_regularizer.scale.requires_grad_(False)
torch.random.manual_seed(0)  # make results deterministic
print(lmbd_initial_guess)

mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
    regularizer=wrapped_regularizer,
    lmbd=lmbd_initial_guess,
    NAG_step_size=1e-2,
    NAG_max_iter=1000,
    NAG_tol=1e-4,
    only_first=only_first,
    device=device,
    verbose=True,
    adaptive_range=True,
)

print("Mean PSNR: ", mean_psnr)
