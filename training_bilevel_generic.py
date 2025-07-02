# Training script for the columns BL+IFT, BL+JFB and ML for CRR, WCRR, ICNN, IDCNN, LAR, TDV and LSR

import torch
from training_methods import bilevel_training, score_training
from dataset import get_dataset
from torchvision.transforms import RandomCrop, CenterCrop, Compose
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
from tqdm import tqdm
import numpy as np
import os

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


problem = "Denoising"  # Denoising or CT
hypergradient_computation = "IFT"  # IFT or JFB
regularizer_name = "WCRR"  # CRR, WCRR, ICNN, IDCNN, LAR, TDV or LSR
load_pretrain = True  # load pretrained weights given that they exist
load_parameter_fitting = (
    True  # load pretrained weights and learned regularization and scaling parameter
)
score_sigma = 3e-2

if regularizer_name == "CRR":
    pretrain_epochs = 300
    pretrain_lr = 1e-2
    fitting_lr = 0.1
    epochs = 100
    lr = 1e-3
    adabelief = True
    jacobian_regularization = True
    jacobian_regularization_parameter = 1e-6
    reg = WCRR(
        sigma=0.1,
        weak_convexity=0.0,
    ).to(device)
elif regularizer_name == "WCRR":
    pretrain_epochs = 300
    pretrain_lr = 1e-2
    fitting_lr = 0.1
    adabelief = True
    epochs = 100
    lr = 1e-3
    jacobian_regularization = True
    jacobian_regularization_parameter = 1e-6
    reg = WCRR(
        sigma=0.1,
        weak_convexity=1.0,
    ).to(device)
elif regularizer_name == "ICNN":
    pretrain_epochs = 300
    pretrain_lr = 1e-3
    fitting_lr = 0.1
    adabelief = True
    epochs = 200
    lr = 1e-3
    jacobian_regularization = True
    jacobian_regularization_parameter = 1e-6
    reg = simple_ICNNPrior(in_channels=1, channels=32, device=device, kernel_size=5).to(
        device
    )
elif regularizer_name == "IDCNN":
    pretrain_epochs = 300
    pretrain_lr = 1e-3
    fitting_lr = 0.01
    adabelief = True
    epochs = 200
    lr = 1e-3
    jacobian_regularization = True
    jacobian_regularization_parameter = 1e-5
    reg = simple_IDCNNPrior(
        in_channels=1, channels=32, device=device, kernel_size=5
    ).to(device)
elif regularizer_name == "LAR":
    pretrain_epochs = 300
    pretrain_lr = 1e-3
    fitting_lr = 0.01
    adabelief = True
    epochs = 200
    lr = 1e-4
    jacobian_regularization = True
    jacobian_regularization_parameter = 1e-5
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
    pretrain_epochs = 7500
    pretrain_lr = 4e-4
    fitting_lr = 0.005
    epochs = 200
    adabelief = True
    lr = 1e-4
    jacobian_regularization_parameter = 1e-4
    jacobian_regularization = True
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
    pretrain_epochs = 7500
    pretrain_lr = 2e-4
    epochs = 200
    adabelief = True
    fitting_lr = 0.05
    lr = 1e-4
    jacobian_regularization = True
    jacobian_regularization_parameter = 1e-4
    reg = LSR(
        nc=[32, 64, 128, 256], pretrained_denoiser=False, alpha=1.0, sigma=score_sigma
    ).to(device)

regularizer = ParameterLearningWrapper(reg, device=device)
lmbd = 1.0

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="log_training_"
    + regularizer_name
    + "_bilevel_"
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

params = 0
for p in regularizer.parameters():
    params += p.numel()
print(params)
logger.info(f"Train {regularizer_name} with {hypergradient_computation} on {problem}")
logger.info(f"The model has {params} parameters")
logger.info(f"Parameters:")
logger.info(
    f"load_pretrain: {load_pretrain}, load_parameter_fitting: {load_parameter_fitting}, score_sigma: {score_sigma}"
)
logger.info(
    f"pretrain_epochs: {pretrain_epochs}, pretrain_lr: {pretrain_lr}, epochs: {epochs}"
)
logger.info(f"adabelief: {adabelief}, fitting_lr: {fitting_lr}, lr: {lr}")
logger.info(
    f"jacobian_regularization: {jacobian_regularization}, jacobian_regularization_parameter: {jacobian_regularization_parameter}, lmbd: {lmbd}"
)


physics, data_fidelity = get_operator(problem, device)

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
    val_set = get_dataset(
        "BSD68"
    )  # torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))

    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=8, shuffle=True, drop_last=True, num_workers=8
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
elif not load_parameter_fitting:
    for p in regularizer.parameters():
        p.requires_grad_(True)
    if regularizer_name == "WCRR":
        regularizer.alpha.requires_grad_(False)
    if regularizer_name == "IDCNN":
        regularizer.alpha.requires_grad_(False)
        regularizer.scale.requires_grad_(False)
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
        sigma=score_sigma,
        epochs=pretrain_epochs,
        lr=pretrain_lr,
        lr_decay=0.1 ** (1 / pretrain_epochs),
        device=device,
        validation_epochs=20,
        logger=logger,
        adabelief=adabelief,
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
    if regularizer_name == "WCRR":
        regularizer.alpha.requires_grad_(False)
        regularizer.regularizer.beta.requires_grad_(True)
    regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
        regularizer,
        physics,
        data_fidelity,
        lmbd,
        train_dataloader,
        val_dataloader,
        epochs=20,
        mode=hypergradient_computation,
        NAG_step_size=1e-1,
        NAG_max_iter=1000,
        NAG_tol_train=1e-4,
        NAG_tol_val=1e-4,
        lr=fitting_lr,
        lr_decay=0.95,
        device=device,
        verbose=False,
        validation_epochs=5,
        logger=logger,
    )
    torch.save(
        regularizer.state_dict(),
        f"weights/score_parameter_fitting_for_{problem}/{regularizer_name}_fitted_parameters_with_{hypergradient_computation}_for_{problem}.pt",
    )

print(regularizer.alpha)
# bilevel training

for p in regularizer.parameters():
    p.requires_grad_(True)
if regularizer_name == "WCRR":
    regularizer.alpha.requires_grad_(False)

if not jacobian_regularization:
    jacobian_regularization_parameter = 0.0

regularizer, loss_train, loss_val, psnr_train, psnr_val = bilevel_training(
    regularizer,
    physics,
    data_fidelity,
    lmbd,
    train_dataloader,
    val_dataloader,
    epochs=epochs,
    mode=hypergradient_computation,
    NAG_step_size=1e-1,
    NAG_max_iter=1500,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    lr=lr,
    lr_decay=0.1 ** (1 / epochs),
    reg=jacobian_regularization,
    reg_para=jacobian_regularization_parameter,
    device=device,
    verbose=False,
    logger=logger,
    adabelief=adabelief,
)

torch.save(
    regularizer.state_dict(),
    f"weights/bilevel_{problem}/{regularizer_name}_bilevel_{hypergradient_computation}_for_{problem}.pt",
)
