import torch
import argparse
from evaluation import evaluate
from dataset import get_dataset
from operators import get_operator
from priors import wcrr

torch.random.manual_seed(0)  # make results deterministic

############################################################

# Define regularizer

parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("-d", "--device", default="cpu", type=str, help="device to use")
device = parser.parse_args().device

weakly = True
pretrained = "weights/WCRR_bilevel.pt" if weakly else "weights/CRR_bilevel.pt"
regularizer = wcrr.WCRR(
    sigma=0.1, weak_convexity=1.0 if weakly else 0.0, pretrained=pretrained
).to(device)

# Parameters for the Nesterov Algorithm, might also be problem dependent...
NAG_step_size = 1 / regularizer.beta.exp().item()  # step size in NAG
NAG_max_iter = 1000  # maximum number of iterations in NAG
NAG_tol = 1e-4  # tolerance for the relative error (stopping criterion)

#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

testing = True
dataset = get_dataset("LoDoPaB", test=testing, transform=None)
physics, data_fidelity = get_operator("CT", device)

# Call unified evaluation routine
if weakly:
    best_lambda = 29.7302
    regularizer.scaling.data = regularizer.scaling + 1.29964928067
else:
    best_lambda = 62.0929
    regularizer.scaling.data = regularizer.scaling + 2.16608472589

mean_psnr, x_out, y_out, recon_out = evaluate(
    physics=physics,
    data_fidelity=data_fidelity,
    dataset=dataset,
    regularizer=regularizer,
    lmbd=best_lambda,
    NAG_step_size=NAG_step_size,
    NAG_max_iter=NAG_max_iter,
    NAG_tol=NAG_tol,
    only_first=False,
    device=device,
    verbose=False,
    adaptive_range=True,
)

print(f"PSNR: {mean_psnr:.4f} dB")
