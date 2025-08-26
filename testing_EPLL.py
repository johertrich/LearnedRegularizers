import torch
from deepinv.optim.utils import GaussianMixtureModel
from evaluation import evaluate
from priors.epll import EPLL
from dataset import get_dataset
from operators import get_evaluation_setting
from pathlib import Path
import os
from torch.utils.data import DataLoader
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.random.manual_seed(0)

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument("--problem", type=str, default="Denoising")
parser.add_argument("--weights", type=str, default=None)
inp = parser.parse_args()

problem = inp.problem

if inp.weights is None:
    weights = problem
else:
    weights = inp.weights

if problem == "Denoising":
    dataset_name = "BSD68"
    transform = None
    lmbd = 13.5
    adaptive_range = False
elif problem == "CT":
    dataset_name = "LoDoPaB"
    transform = None
    lmbd = 500.0
    adaptive_range = True
else:
    raise NotImplementedError("Problem not found")

# Get the test dataset
print(f"Loading test dataset: {dataset_name}")
test_dataset = get_dataset(dataset_name, test=True, transform=None)
channels = test_dataset[0].shape[0]

dataset, physics, data_fidelity = get_evaluation_setting(problem, device)

weights_filepath = f"weights/gmm_{weights}.pt"
print(f"Loading GMM weights from {weights_filepath}")

filename = Path(weights_filepath)

if filename.is_file():
    print("GMM weights file exists. Loading GMM weights")
    setup_data = torch.load(weights_filepath)
else:
    raise FileNotFoundError("GMM weights file does not exist")


patch_size = setup_data["patch_size"]
n_gmm_components = setup_data["n_gmm_components"]

GMM = GaussianMixtureModel(n_gmm_components, patch_size**2 * channels, device=device)
GMM.load_state_dict(setup_data["weights"])

regularizer = EPLL(
    device=device,
    patch_size=patch_size,
    channels=channels,
    n_gmm_components=n_gmm_components,
    GMM=GMM,
    pad=True,
    batch_size=30000,
)

print(f"Testing for {problem} with {weights} weights")
print("-" * 30)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

save_dir = f"results/EPLL_{problem}problem_{weights}weights"
os.makedirs(save_dir, exist_ok=True)

mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=test_dataset,
    regularizer=regularizer,
    lmbd=lmbd,
    NAG_step_size=1e-3,
    NAG_max_iter=1000,
    NAG_tol=1e-4,
    adam=True,
    only_first=False,
    device=device,
    verbose=False,
    save_path=save_dir,
    save_png=True,
    adaptive_range=adaptive_range,
)

data = {"Mean PSNR": float(mean_psnr), "lmbd": lmbd, "gmm_weights": weights_filepath}
with open(os.path.join(save_dir, "results.yaml"), "w") as f:
    yaml.dump(data, f)
