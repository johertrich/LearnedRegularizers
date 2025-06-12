"""Evaluate LPN for denoising."""

from operators import get_evaluation_setting
from deepinv.utils.plotting import plot
from evaluation import evaluate
from dataset import get_dataset
from torchvision.transforms import CenterCrop
import torch
from priors.lpn.lpn import LPNPrior
from torch.utils.data import DataLoader
from deepinv.loss.metric import PSNR
from training_methods.simple_lpn_training import lpn_denoise_patch
import numpy as np

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
pretrained = "weights/lpn_64_neg1_pm/simple_LPN.pt"
regularizer = LPNPrior(pretrained=pretrained).to(device)


#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)


# Call lpn evaluation routine
def evaluate(
    physics,
    data_fidelity,
    dataset,
    regularizer,
    only_first=False,
    adaptive_range=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
):

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    if adaptive_range:
        psnr = PSNR(max_pixel=None)
    else:
        psnr = PSNR()

    regularizer.eval()
    for p in regularizer.parameters():
        p.requires_grad_(False)

    ## Evaluate on the test set
    psnrs = []
    for i, x in enumerate(dataloader):
        if device == "mps":
            # mps does not support float64
            x = x.to(torch.float32).to(device)
        else:
            x = x.to(device).to(torch.float32)
        y = physics(x)

        # denoise using LPN
        recon = lpn_denoise_patch(
            y, regularizer.lpn, regularizer.lpn.img_size, regularizer.lpn.img_size // 2
        )

        psnrs.append(psnr(recon, x).squeeze().item())
        if i == 0:
            y_out = y
            x_out = x
            recon_out = recon
        if only_first:
            break
    mean_psnr = np.mean(psnrs)
    print("Mean PSNR over the test set: {0:.2f}".format(mean_psnr))
    return mean_psnr, x_out, y_out, recon_out


mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
    regularizer=regularizer,
    only_first=only_first,
    device=device,
)

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
