from priors import (
    ParameterLearningWrapper,
    WCRR,
    ICNNPrior,
    IDCNNPrior,
    LSR,
    TDV,
    LocalAR,
    EPLL,
    PatchNR,
)
from training_methods import bilevel_training
from deepinv.optim.utils import GaussianMixtureModel
import torch
from operators import get_evaluation_setting
from dataset import get_dataset
from evaluation import evaluate
import argparse
import os
import logging
import datetime
import numpy as np

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
lr = 0.1

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
    if evaluation_mode == "AR":
        output_factor = (
            362 ** 2 / 321 ** 2
        )  # due to the mean reduction the regularization constant must be adapted for different image sizes
        reg = LocalAR(
            in_channels=1,
            pad=True,
            use_bias=True,
            n_patches=-1,
            output_factor=output_factor,
        ).to(device)
    else:
        reg = LocalAR(
            in_channels=1,
            pad=True,
            use_bias=False,
            n_patches=-1,
            reduction="sum",
            output_factor=1 / 142 ** 2,
            pretrained=None,
        ).to(device)
elif regularizer_name == "TDV":
    config = dict(
        in_channels=1,
        num_features=32,
        multiplier=1,
        num_mb=3,
        num_scales=3,
        zero_mean=True,
    )
    reg = TDV(**config).to(device)
elif regularizer_name == "LSR":
    reg = LSR(
        nc=[32, 64, 128, 256], pretrained_denoiser=False, alpha=1.0, sigma=3e-2
    ).to(device)
elif regularizer_name == "EPLL":
    weights_filepath = f"weights/gmm_Denoising.pt"
    setup_data = torch.load(weights_filepath)
    patch_size = setup_data["patch_size"]
    n_gmm_components = setup_data["n_gmm_components"]
    GMM = GaussianMixtureModel(n_gmm_components, patch_size ** 2, device=device)
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
    lmbd = 500.0
    lmbd_guesses = [0.8 * lmbd + i * (0.4 * lmbd) / 9 for i in range(10)]
elif regularizer_name == "PatchNR":
    weights_filepath = f"weights/patchnr/patchnr_6x6_BSD500_fitted.pt"
    weights = torch.load(weights_filepath, map_location=device)
    regularizer = PatchNR(
        patch_size=6,
        channels=1,
        num_layers=5,
        sub_net_size=weights["patchnr_subnetsize"],
        device=device,
        n_patches=weights["n_patches"],
        pretrained=None,
        pad=True,
    )
    regularizer.load_state_dict(weights["weights"])
    min_lmbd = 280.0
    max_lmbd = 320.0
    lmbd_guesses = np.linspace(min_lmbd, max_lmbd, 15)


if regularizer_name in ["EPLL", "PatchNR"]:
    pass
elif (
    evaluation_mode == "bilevel-IFT"
    or evaluation_mode == "bilevel-JFB"
    or evaluation_mode == "Score"
):
    if regularizer_name == "IDCNN":
        mode = "JFB"
    if regularizer_name == "LSR":
        lr = 0.1
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
    """
    # Shall be removed after having run training_AR.py
    if regularizer_name == "IDCNN":
        pretrained = "./weights/simple_simple_IDCNNPrior_ar_Denoising.pt"
        regularizer.load_state_dict(torch.load(pretrained, map_location=device))
    elif regularizer_name == "ICNN":
        pretrained = "./weights/simple_simple_ICNNPrior_ar_Denoising.pt"
        cpk = torch.load(pretrained, map_location=device)
        regularizer.load_state_dict(cpk)
    else:
        raise ValueError(f"no configuration for AR with regularizer {regularizer_name}")"""
    regularizer = ParameterLearningWrapper(reg, device=device)
    weights = torch.load(
        f"weights/adversarial_Denoising/{regularizer_name}_adversarial_for_Denoising_fitted.pt",
        map_location=device,
    )
    regularizer.load_state_dict(weights)
else:
    raise ValueError("Unknown evaluation mode!")

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="log_parameter_fitting_"
    + regularizer_name
    + "_"
    + evaluation_mode
    + "_"
    + str(datetime.datetime.now())
    + ".log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

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
    if regularizer_name == "EPLL":
        setup_data = torch.load(f"weights/Denoising_to_CT/EPLL", map_location=device)
        regularizer.GMM.load_state_dict(setup_data["weights"])
        lmbd_initial_guess = setup_data["lambda"]
        wrapped_regularizer = regularizer
    elif regularizer_name == "PatchNR":
        weights = torch.load(f"weights/Denoising_to_CT/PatchNR", map_location=device)
        regularizer.load_state_dict(weights["weights"])
        lmbd_initial_guess = weights["lambda"]
        wrapped_regularizer = regularizer
    else:
        wrapped_regularizer.load_state_dict(
            torch.load(
                f"weights/Denoising_to_CT/{regularizer_name}_with_{evaluation_mode}",
                map_location=device,
            )
        )
elif regularizer_name in ["EPLL", "PatchNR"]:
    best_lmbd = -1
    best_psnr = -999
    for lmbd in lmbd_guesses:
        print(lmbd)
        mean_psnr, x_out, y_out, recon_out = evaluate(
            physics=physics,
            data_fidelity=data_fidelity,
            dataset=validation_dataset,
            regularizer=regularizer,
            lmbd=lmbd,
            NAG_step_size=1e-3,
            NAG_max_iter=3000,
            NAG_tol=1e-4,
            only_first=False,
            adaptive_range=True,
            device=device,
            adam=True,
            logger=logger,
        )
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            best_lmbd = lmbd
    lmbd_initial_guess = best_lmbd
    wrapped_regularizer = regularizer
    if regularizer_name == "EPLL":
        setup_data["lambda"] = best_lmbd
        torch.save(setup_data, f"weights/Denoising_to_CT/EPLL")
    elif regularizer_name == "PatchNR":
        weights["lambda"] = best_lmbd
        torch.save(weights, f"weights/Denoising_to_CT/PatchNR")
else:
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
        momentum_optim=(0.5, 0.9),
        lr_decay=0.999,
        device=device,
        verbose=False,
        validation_epochs=100,
        dynamic_range_psnr=True,
        logger=logger,
    )
    torch.save(
        wrapped_regularizer.state_dict(),
        f"weights/Denoising_to_CT/{regularizer_name}_with_{evaluation_mode}",
    )

if regularizer_name not in ["EPLL", "PatchNR"]:
    print("Final alpha: ", wrapped_regularizer.alpha)
    print("Final scale: ", wrapped_regularizer.scale)

    wrapped_regularizer.alpha.requires_grad_(False)
    wrapped_regularizer.scale.requires_grad_(False)

only_first = False
torch.random.manual_seed(0)  # make results deterministic
print(lmbd_initial_guess)

mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
    regularizer=wrapped_regularizer,
    lmbd=lmbd_initial_guess,
    NAG_step_size=1e-3 if regularizer_name in ["EPLL", "PatchNR"] else 1e-2,
    NAG_max_iter=3000 if regularizer_name in ["EPLL", "PatchNR"] else 1000,
    NAG_tol=1e-4,
    only_first=only_first,
    device=device,
    verbose=True,
    adam=regularizer_name in ["EPLL", "PatchNR"],
    adaptive_range=True,
)

print("Mean PSNR: ", mean_psnr)

logger.info("Mean PSNR: " + str(mean_psnr))
