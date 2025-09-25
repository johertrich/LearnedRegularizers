import torch
from deepinv.datasets import PatchDataset
from torchvision.transforms import CenterCrop
from deepinv.optim.utils import GaussianMixtureModel
from evaluation import evaluate
from priors.epll import EPLL
from dataset import get_dataset
from operators import get_operator
from pathlib import Path
import os
import argparse
import logging
import datetime

torch.random.manual_seed(0)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument("--problem", type=str, default="Denoising")
inp = parser.parse_args()

problem = inp.problem

if problem == "Denoising":
    dataset_name = "BSDS500_gray"
    transform = CenterCrop(321)
    lmbd = 14.0
    adaptive_range = False
elif problem == "CT":
    dataset_name = "LoDoPaB"
    transform = None
    lmbd = 500.0
    adaptive_range = True
else:
    raise NotImplementedError("Problem not found")

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="log_training_"
    + problem
    + "_EPLL_"
    + str(datetime.datetime.now())
    + ".log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)
train_dataset = get_dataset(dataset_name, test=False, transform=transform)
val_dataset = get_dataset(dataset_name, test=False, transform=None)
physics, data_fidelity = get_operator(problem, device)

# Split the full train into training and validation set. The training is used to learn the GMM weights
val_ratio = 0.001 if problem == "CT" else 0.03
val_len = int(len(train_dataset) * val_ratio)
train_len = len(train_dataset) - val_len
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = torch.utils.data.Subset(val_dataset, range(train_len, len(train_dataset)))
channels = train_dataset[0].shape[0]

train_imgs = []
num_training_images = 100
for i in range(num_training_images):
    train_imgs.append(train_set[i].unsqueeze(0).float())
train_imgs = torch.concat(train_imgs)
channels = train_imgs.shape[1]

val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=1, shuffle=False, drop_last=True
)

patch_sizes = [6]
n_gmm_components = [300, 400]
best_mean_psnr = -float("inf")

for j, patch_size in enumerate(patch_sizes):

    train_patch_dataset = PatchDataset(
        train_imgs, patch_size=patch_size, transforms=None
    )

    patch_dataloader = torch.utils.data.DataLoader(
        train_patch_dataset,
        batch_size=1024,
        shuffle=True,
        drop_last=True,
    )
    for k, n_gmm_component in enumerate(n_gmm_components):
        logger.info("-" * 30)
        logger.info(
            f"Running for patch size {patch_size} and {n_gmm_component} components"
        )

        GMM = GaussianMixtureModel(
            n_gmm_component, patch_size**2 * channels, device=device
        )
        GMM.fit(patch_dataloader, verbose=True, max_iters=50, stopping_criterion=1e-4)
        logger.info("Fitting GMM done")

        # Create the EPLL regularizer with the learned GMM
        regularizer = EPLL(
            device=device,
            patch_size=patch_size,
            channels=channels,
            n_gmm_components=n_gmm_component,
            GMM=GMM,
            pad=True,
            batch_size=30000,
        )

        # Reconstruction with the given GMM
        mean_psnr, x_out, y_out, recon_out = evaluate(
            physics=physics,
            data_fidelity=data_fidelity,
            dataset=val_set,
            regularizer=regularizer,
            lmbd=lmbd,
            step_size=1e-3,
            max_iter=1000,
            tol=1e-4,
            adam=True,
            only_first=False,
            device=device,
            verbose=False,
            adaptive_range=adaptive_range,
        )
        logger.info(
            f"Mean PSNR for patch size {patch_size} and {n_gmm_component} components: {mean_psnr:.2f}"
        )
        if mean_psnr > best_mean_psnr:
            best_mean_psnr = mean_psnr
            best_gmm = GMM.state_dict()
            best_patch_size = patch_size
            best_n_gmm_component = n_gmm_component
            logger.info(f"\t New best GMM found!")

logger.info(
    f"Best GMM: Patch Size: {best_patch_size}, Components: {best_n_gmm_component}, Mean PSNR: {best_mean_psnr:.2f}"
)

# Fine tune lambda for the best fitted model with current lambda estimate and current best model
best_lamb = lmbd
GMM = GaussianMixtureModel(
    best_n_gmm_component, best_patch_size**2 * channels, device=device
)
GMM.load_state_dict(best_gmm)
regularizer = EPLL(
    device=device,
    patch_size=best_patch_size,
    channels=channels,
    n_gmm_components=best_n_gmm_component,
    GMM=GMM,
    pad=True,
    batch_size=30000,
)

for lamb in [0.8 * lmbd + i * (0.4 * lmbd) / 9 for i in range(10)]:
    mean_psnr, x_out, y_out, recon_out = evaluate(
        physics=physics,
        data_fidelity=data_fidelity,
        dataset=val_set,
        regularizer=regularizer,
        lmbd=lamb,
        step_size=1e-3,
        max_iter=1000,
        tol=1e-4,
        adam=True,
        only_first=False,
        device=device,
        verbose=False,
        adaptive_range=adaptive_range,
    )
    logger.info(f"Mean PSNR for lambda {lamb}: {mean_psnr:.2f}")
    if mean_psnr > best_mean_psnr:
        best_mean_psnr = mean_psnr
        best_lamb = lamb

logger.info(f"Best lambda: {best_lamb}")

gmm_dir = Path(f"weights")
gmm_dir.mkdir(parents=True, exist_ok=True)
gmm_filepath = gmm_dir / "gmm_{}.pt".format(problem)

data = {
    "patch_size": best_patch_size,
    "n_gmm_components": best_n_gmm_component,
    "gmm_weights": str(gmm_filepath),
    "training mean psnr": float(best_mean_psnr),
    "weights": best_gmm,
    "lambda": best_lamb,
}
torch.save(data, gmm_filepath)
