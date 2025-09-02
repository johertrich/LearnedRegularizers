# -*- coding: utf-8 -*-
"""
Training of Local (patch-based) adversarial regulariser.

@author: Alexander
"""

from priors import LocalAR, ParameterLearningWrapper
import torch
from training_methods import ar_training, bilevel_training
from training_methods.ar_training import estimate_lmbd, estimate_lip
from dataset import get_dataset

from torchvision.transforms import RandomCrop
from operators import get_operator
import argparse
import os 

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument("--problem", type=str, default="Denoising", choices=["Denoising", "CT"])
parser.add_argument("--only_fitting", type=bool, default=False)

inp = parser.parse_args()

problem = inp.problem
only_fitting = inp.only_fitting  

if not os.path.isdir("weights"):
    os.mkdir("weights")
if not os.path.isdir("weights/local_adversarial"):
    os.mkdir("weights/local_adversarial")

# problem dependent parameters
if problem == "Denoising":
    physics , data_fidelity = get_operator(problem, device)
    dataset = get_dataset("BSDS500_gray", test=False, transform=RandomCrop(128))
    lmbd = 10.0
    test_ratio = 0.1
elif problem == "CT":
    dataset = get_dataset("LoDoPaB", test=False, transform=None)
    physics , data_fidelity = get_operator(problem, device)
    lmbd = 1.0
    test_ratio = 0.003
else:
    raise NotImplementedError

# splitting in training and validation set
test_len = int(len(dataset) * test_ratio)
train_len = len(dataset) - test_len
train_set, val_set = torch.utils.data.random_split(dataset, [train_len, test_len])

fitting_set = torch.utils.data.Subset(val_set, range(0, 5))
fitting_dataloader = torch.utils.data.DataLoader(
    fitting_set, batch_size=5, shuffle=True, drop_last=True
)

print("train set: ", len(train_set))
print("val set: ", len(val_set))

batch_size = 256
shuffle = True

patch_size = 15

regularizer = LocalAR(in_channels=1, pad=True, use_bias=True, n_patches=-1).to(device)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=12, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1)
lmbd = estimate_lmbd(train_loader, physics, device="cuda").item()

print("estimated : ", lmbd)
dataset_name = "BSD500" if problem == "Denoising" else "LoDoPab"

if only_fitting:
    ckp = torch.load(
        f"weights/local_adversarial/{regularizer.__class__.__name__}_adversarial_p={patch_size}x{patch_size}_{dataset_name}.pt"
    )
    regularizer.cnn.load_state_dict(ckp)
else:

    mu = 5.0
    ar_training(
        regularizer=regularizer,
        physics=physics,
        data_fidelity=data_fidelity,
        lmbd=lmbd,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        patch_size=patch_size,
        device=device,
        epochs=100,
        lr=1e-3,
        mu=mu,
        savestr=None,
        validation_epochs=5,
        dynamic_range_psnr = True if problem == "CT" else False,
        patches_per_img=64,
    )


    torch.save(
        regularizer.state_dict(),
        f"weights/local_adversarial/{regularizer.__class__.__name__}_adversarial_p={patch_size}x{patch_size}_{dataset_name}.pt",
    )


lmbd = estimate_lmbd(val_loader, physics, device)
lip = estimate_lip(regularizer, val_loader, device)
lmbd_est = lmbd / lip
print("Estimated Lambda for Fitting:", lmbd_est)

wrapped_regularizer = ParameterLearningWrapper(regularizer, device=device)
with torch.no_grad():
    wrapped_regularizer.alpha.copy_(torch.log(lmbd_est))

for p in wrapped_regularizer.parameters():
    p.requires_grad_(False)
    wrapped_regularizer.alpha.requires_grad_(True)
    wrapped_regularizer.scale.requires_grad_(True)

# parameter search on first five images of training set

bilevel_training(
    wrapped_regularizer,
    physics,
    data_fidelity,
    1,
    fitting_dataloader,
    val_loader,
    epochs=100,
    mode="IFT",
    NAG_step_size=1e-1,
    NAG_max_iter=1000,
    NAG_tol_train=1e-4,
    NAG_tol_val=1e-4,
    minres_tol=1e-4,
    lr=1e-3,
    lr_decay=0.98 if problem == "CT" else 0.99,
    reg=False if problem == "CT" else True,
    device=device,
    verbose=True,
    validation_epochs=10,
    dynamic_range_psnr=True if problem == "CT" else False,
    adabelief=True,
)

torch.save(
    wrapped_regularizer.state_dict(),
    f"weights/adversarial_{problem}/{regularizer.__class__.__name__}_adversarial_for_{problem}_fitted.pt",
)

print("Final alpha: ", torch.exp(wrapped_regularizer.alpha))
print("Final scale: ", wrapped_regularizer.scale)
