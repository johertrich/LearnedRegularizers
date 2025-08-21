# Training script for the columns BL+IFT, BL+JFB and ML for CRR, WCRR, ICNN, IDCNN, LAR, TDV and LSR

import torch
from torch.utils.data import Subset
from training_methods import bilevel_training, score_training, bilevel_training_maid
from dataset import get_dataset, PatchesDataset
from priors import (
    ParameterLearningWrapper,
    WCRR,
    simple_ICNNPrior,
    simple_IDCNNPrior,
    LSR,
    TDV,
    LocalAR,
)
from torchvision.transforms import (
    RandomCrop,
    RandomVerticalFlip,
    Compose,
    RandomHorizontalFlip,
    CenterCrop,
    RandomApply,
    RandomRotation,
)
from operators import get_operator
import logging
import datetime
import numpy as np
import os
from hyperparameters import get_bilevel_hyperparameters
import argparse

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument("--problem", type=str, default="Denoising")
parser.add_argument("--hypergradient", type=str, default="IFT")
parser.add_argument("--MAID", type=bool, default=False)
parser.add_argument("--regularizer_name", type=str, default="CRR")
parser.add_argument("--load_pretrain", type=bool, default=False)
parser.add_argument("--load_parameter_fitting", type=bool, default=False)
inp = parser.parse_args()

problem = inp.problem  # Denoising or CT
hypergradient_computation = inp.hypergradient  # IFT or JFB
MAID = inp.MAID # Using MAID training or not
regularizer_name = inp.regularizer_name  # CRR, WCRR, ICNN, IDCNN, LAR, TDV or LSR
load_pretrain = inp.load_pretrain  # load pretrained weights given that they exist
load_parameter_fitting = (
    inp.load_parameter_fitting
)  # load pretrained weights and learned regularization and scaling parameter


hyper_params = get_bilevel_hyperparameters(regularizer_name, problem)

# Choose Regularizer
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
        nc=[32, 64, 128, 256],
        pretrained_denoiser=False,
        alpha=1.0,
        sigma=hyper_params.score_sigma,
    ).to(device)

regularizer = ParameterLearningWrapper(reg, device=device)
lmbd = 1.0

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="log_training_"
    + problem
    + "_"
    + regularizer_name
    + "_bilevel_" if not MAID else "_bilevel_MAID_"
    + hypergradient_computation
    + "_"
    + str(datetime.datetime.now())
    + ".log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)
if not os.path.isdir("weights"):
    os.mkdir("weights")
if not os.path.isdir(f"weights/score_for_{problem}"):
    os.mkdir(f"weights/score_for_{problem}")
if not os.path.isdir(f"weights/score_parameter_fitting_for_{problem}"):
    os.mkdir(f"weights/score_parameter_fitting_for_{problem}")
if not os.path.isdir(f"weights/bilevel_{problem}"):
    os.mkdir(f"weights/bilevel_{problem}")
if not os.path.isdir(f"weights/bilevel_MAID_{problem}"):
    os.mkdir(f"weights/bilevel_MAID_{problem}")
params = 0
for p in regularizer.parameters():
    params += p.numel()
print(params)
if not MAID:
    logger.info(f"Train {regularizer_name} with {hypergradient_computation} on {problem}")
else:
    logger.info(f"Train {regularizer_name} with {hypergradient_computation} MAID on {problem}")
logger.info(f"The model has {params} parameters")
logger.info("Parameters:")
logger.info(
    f"load_pretrain: {load_pretrain}, load_parameter_fitting: {load_parameter_fitting}, score_sigma: {hyper_params.score_sigma}"
)
logger.info(
    f"pretrain_epochs: {hyper_params.pretrain_epochs}, pretrain_lr: {hyper_params.pretrain_lr}, epochs: {hyper_params.epochs}"
)
logger.info(
    f"adabelief: {hyper_params.adabelief}, fitting_lr: {hyper_params.fitting_lr}, lr: {hyper_params.lr}"
)
logger.info(
    f"jacobian_regularization: {hyper_params.jacobian_regularization}, jacobian_regularization_parameter: {hyper_params.jacobian_regularization_parameter}, lmbd: {lmbd}"
)

# The noise model in MAID should be deterministic not changing at each call
physics, data_fidelity = get_operator(problem, device, MAID=MAID)

rotation_flip_transform = Compose(
    [
        RandomCrop(128),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomApply([RandomRotation((90, 90))], p=0.5),
    ]
)

if problem == "Denoising":
    train_dataset = get_dataset(
        "BSDS500_gray", test=False, transform=rotation_flip_transform
    )
    val_dataset = get_dataset("BSDS500_gray", test=False, transform=CenterCrop(321))
    # splitting in training and validation set
    test_ratio = 0.1
    test_len = int(len(train_dataset) * 0.1)
    train_len = len(train_dataset) - test_len
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    pretrain_dataset = train_set
    val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))
    train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=8, shuffle=True, drop_last=True, num_workers=8
    )
    fitting_dataloader = train_dataloader
    if MAID:
        # Define patch parameters
        PATCH_SIZE = 64
        STRIDE = 64  # Use PATCH_SIZE for non-overlapping patches
        SUBSET = 32
        patch_dataset = PatchesDataset(
        original_dataset=train_dataset,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        transform=None,  # Add post-patch transforms here if needed
        )
        num_patches_total = len(patch_dataset)
        num_subset_patches = SUBSET
        # Use a fixed range for deterministic subset selection
        subset_indices = torch.randint(0, num_patches_total, (num_subset_patches,))
        train_subset = Subset(patch_dataset, subset_indices)
        train_dataloader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=num_subset_patches,
            shuffle=False,
            drop_last=True,
            num_workers=8,
        )
elif problem == "CT":
    train_dataset = get_dataset("LoDoPaB", test=False)
    pretrain_dataset = get_dataset(
        "LoDoPaB", test=False, transform=rotation_flip_transform
    )
    val_dataset = get_dataset("LoDoPaB", test=False)
    # splitting in training and validation set
    test_ratio = 0.1
    test_len = int(len(train_dataset) * 0.1)
    train_len = len(train_dataset) - test_len
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=8, shuffle=True, drop_last=True, num_workers=8
    )
    fitting_set = get_dataset("LoDoPaB_val")
    # use smaller dataset for parameter fitting
    fitting_dataloader = torch.utils.data.DataLoader(
        fitting_set, batch_size=5, shuffle=True, drop_last=True, num_workers=8
    )


val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True, num_workers=8
)
pretrain_dataloader = torch.utils.data.DataLoader(
    pretrain_dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=8
)

if load_pretrain and not load_parameter_fitting:
    regularizer.load_state_dict(
        torch.load(
            f"weights/score_for_{problem}/{regularizer_name}_score_training_for_{problem}.pt"
        )
    )
elif not load_parameter_fitting and not hyper_params.pretrain_epochs == 0:
    for p in regularizer.parameters():
        p.requires_grad_(True)
    if (
        regularizer_name == "IDCNN" and problem == "CT"
    ):  # to ensure that the variational problem has a solution after pretraining, we enforce convexity of the IDCNN in the pretraining
        regularizer.regularizer.icnn2.wz.weight.data.fill_(0)
        regularizer.regularizer.icnn2.wz.weight.requires_grad_(False)
    regularizer.alpha.requires_grad_(hyper_params.pretrain_alpha)
    regularizer.scale.requires_grad_(hyper_params.pretrain_scale)
    (
        regularizer,
        loss_train,
        loss_val,
        psnr_train,
        psnr_val,
    ) = score_training.score_training(
        regularizer,
        pretrain_dataloader,
        val_dataloader,
        sigma=hyper_params.score_sigma,
        epochs=hyper_params.pretrain_epochs,
        lr=hyper_params.pretrain_lr,
        weight_decay=hyper_params.pretrain_weight_decay,
        lr_decay=0.1 ** (1 / hyper_params.pretrain_epochs),
        device=device,
        validation_epochs=20,
        logger=logger,
        adabelief=hyper_params.adabelief,
        dynamic_range_psnr=problem == "CT",
        model_selection=False,
        # loss_fn=lambda x,y:torch.abs(x-y).sum()
    )
    torch.save(
        regularizer.state_dict(),
        f"weights/score_for_{problem}/{regularizer_name}_score_training_for_{problem}.pt",
    )

if load_parameter_fitting:
    regularizer.load_state_dict(
        torch.load(
            f"weights/score_parameter_fitting_for_{problem}/{regularizer_name}_fitted_parameters_with_{hypergradient_computation}_for_{problem}.pt"
        )
    )
else:
    for p in regularizer.parameters():
        p.requires_grad_(False)
    regularizer.alpha.requires_grad_(True)
    regularizer.scale.requires_grad_(True)
    regularizer.alpha.data = regularizer.alpha.data + np.log(
        hyper_params.parameter_fitting_init
    )

    if (
        regularizer_name == "WCRR" and problem == "Denoising"
    ):  # don't tune the regularization parameter for the WCRR to ensure 1-weak convexity, use beta instead
        regularizer.alpha.requires_grad_(False)
        regularizer.regularizer.beta.requires_grad_(True)
    if hyper_params.do_parameter_fitting:
        regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
            regularizer,
            physics,
            data_fidelity,
            lmbd,
            fitting_dataloader,
            val_dataloader,
            epochs=20 if problem == "Denoising" else 100,
            mode=hypergradient_computation,
            NAG_step_size=1e-1,
            NAG_max_iter=1500,
            NAG_tol_train=1e-4,
            NAG_tol_val=1e-4,
            lr=hyper_params.fitting_lr,
            momentum_optim=(0.5, 0.9),
            reg=False,  # jacobian_regularization,
            reg_para=hyper_params.jacobian_regularization_parameter,
            reg_reduced=problem == "CT",
            lr_decay=0.95,
            device=device,
            verbose=False,
            dynamic_range_psnr=problem == "CT",
            validation_epochs=5 if problem == "Denoising" else 25,
            logger=logger,
        )
    torch.save(
        regularizer.state_dict(),
        f"weights/score_parameter_fitting_for_{problem}/{regularizer_name}_fitted_parameters_with_{hypergradient_computation}_for_{problem}.pt",
    )

# bilevel training

for p in regularizer.parameters():
    p.requires_grad_(True)
if regularizer_name == "WCRR":  # fix regularization parameter to ensure 1-weak convexit
    regularizer.alpha.requires_grad_(False)
if not MAID:
    regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
        regularizer,
        physics,
        data_fidelity,
        lmbd,
        train_dataloader,
        val_dataloader,
        epochs=hyper_params.epochs,
        mode=hypergradient_computation,
        NAG_step_size=1e-1,
        NAG_max_iter=1500,
        NAG_tol_train=1e-4,
        NAG_tol_val=1e-4,
        lr=hyper_params.lr,
        lr_decay=0.1 ** (1 / hyper_params.epochs),
        reg=hyper_params.jacobian_regularization,
        reg_para=hyper_params.jacobian_regularization_parameter,
        reg_reduced=problem == "CT",
        device=device,
        verbose=False,
        logger=logger,
        adabelief=hyper_params.adabelief,
        dynamic_range_psnr=problem == "CT",
        validation_epochs=20 if problem == "Denoising" else 1,
    )
    torch.save(
    regularizer.state_dict(),
    f"weights/bilevel_{problem}/{regularizer_name}_bilevel_{hypergradient_computation}_for_{problem}.pt",
    )
else:
    #hyperparameters of MAID
    eps = 1e-1
    alpha = 1e-1
    regularizer, loss_train, loss_val, psnr_train, psnr_val, _, _, logs, _ = (
        bilevel_training_maid(
            regularizer,
            physics,
            data_fidelity,
            lmbd,
            train_dataloader,
            val_dataloader,
            epochs=300,
            NAG_step_size=1e-1,
            NAG_max_iter=1000,
            NAG_tol_train=eps,
            NAG_tol_val=1e-4,
            CG_tol=eps,
            lr=alpha,
            lr_decay=0.25,
            device=device,
            precondition=True,  # Use preconditioned upper-level optimization (AdaGrad)
            verbose=True,
            save_dir=str(SUBSET)
            + "_"
            + str(eps)
            + "_"
            + str(alpha)
            + "_CRR_MAID",  # Directory to save the model and logs
            algorithm = "MAID Adagrad",  # Algorithm used for training
        )
    )
    torch.save(
    regularizer.state_dict(),
    f"weights/bilevel_{problem}/{regularizer_name}_bilevel_MAID_{hypergradient_computation}_for_{problem}.pt",
    )


