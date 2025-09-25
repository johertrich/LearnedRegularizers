# Evaluates a TV baseline on the test problems. Consequently it does not use the
# NAG-based evaluation routine but rather uses a primal-dual hybrid gradient algorithm.

from operators import get_evaluation_setting
from deepinv.utils.plotting import plot
import torch
from deepinv.models import TVDenoiser
from deepinv.physics import LinearPhysics
from deepinv.optim import L1Prior
import numpy as np
from torch.utils.data import DataLoader
from deepinv.loss.metric import PSNR
from tqdm import tqdm
from dataset import get_dataset
import argparse

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device: ", device)
torch.random.manual_seed(0)  # make results deterministic

############################################################

# Problem selection
parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument("--problem", type=str, default="Denoising")
inp = parser.parse_args()

problem = inp.problem  # Select problem setups, which we consider.
only_first = False  # just evaluate on the first image of the dataset for test purposes

############################################################

# reconstruction hyperparameters, might be problem dependent
if problem == "CT":
    lmbd_choices = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]  # choices to test for the regularization parameter
    step_size = 0.005  # step sizes in the PDHG
    sigma_tau_factor = 1e3
elif problem == "Denoising":
    lmbd_choices = [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1]
    step_size = 0.2
    sigma_tau_factor = 1

#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)

# validation set for parameter fitting
if problem == "CT":
    validation_dataset = get_dataset("LoDoPaB_val")
elif problem == "Denoising":
    validation_dataset = torch.utils.data.Subset(
        get_dataset("BSDS500_gray", test=False), range(5)
    )


class ParallelPrimalDualOptimizer:
    def __init__(
        self,
        physics_list,
        prox_list,
        lambda_list,
        steps,
        sigma,
        tau,
        theta=1.0,
        stopping_criterion=1e-5,
    ):
        self.physics_list = physics_list
        self.prox_list = prox_list
        self.lambda_list = lambda_list
        self.steps = steps
        self.sigma = sigma
        self.tau = tau
        self.theta = theta
        self.stopping_criterion = stopping_criterion

    def optimize(self, x_init):
        x = x_init
        for step in range(self.steps):
            x_old = x.clone()
            if step == 0:
                args = [physics.A(x) for physics in self.physics_list]
                bs = [torch.zeros_like(arg) for arg in args]
            else:
                args = [b + physics.A(x) for (b, physics) in zip(bs, self.physics_list)]
            ys = [
                prox(arg, gamma=lamb / self.sigma)
                for (arg, prox, lamb) in zip(args, self.prox_list, self.lambda_list)
            ]
            bs_old = [b.clone() for b in bs]
            bs = [
                b + physics.A(x) - y
                for (b, y, physics) in zip(bs, ys, self.physics_list)
            ]
            bs_tilde = [
                (1 + self.theta) * b - self.theta * b_old
                for (b, b_old) in zip(bs, bs_old)
            ]
            x = x - self.tau * self.sigma * torch.sum(
                torch.stack(
                    [
                        physics.A_adjoint(b_tilde)
                        for b_tilde, physics in zip(bs_tilde, self.physics_list)
                    ],
                    0,
                ),
                0,
            )
            rel_err = torch.linalg.norm(
                x_old.flatten() - x.flatten()
            ) / torch.linalg.norm(x.flatten() + 1e-12)
            if rel_err < self.stopping_criterion:
                break
        return x


def eval_TV(lmbd, dataset, plot_example=False):
    TVPhysics = LinearPhysics(A=TVDenoiser.nabla, A_adjoint=TVDenoiser.nabla_adjoint)
    L1_Prior = L1Prior()

    physics_list = [physics, TVPhysics]
    lambda_list = [1.0, lmbd]
    prox_list = [
        lambda u, gamma: data_fidelity.prox_d(u, y, gamma=gamma),
        L1_Prior.prox,
    ]

    optimizer = ParallelPrimalDualOptimizer(
        physics_list,
        prox_list,
        lambda_list,
        20000,
        step_size * sigma_tau_factor,
        step_size / sigma_tau_factor,
        stopping_criterion=1e-5,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    if problem == "CT":
        psnr = PSNR(max_pixel=None)
        dagger_name = "FBP"
    elif problem == "Denoising":
        psnr = PSNR()
        dagger_name = "Noisy"

    psnrs = []
    psnrs_dagger = []
    for i, x in (progress_bar := tqdm(enumerate(dataloader))):
        if device == "mps":
            x = x.to(torch.float32).to(device)
        else:
            x = x.to(device).to(torch.float32)
        y = physics(x)
        x_init = physics.A_dagger(y)
        psnrs_dagger.append(psnr(x_init, x).squeeze().item())
        recon = optimizer.optimize(x_init)
        psnrs.append(psnr(recon, x).squeeze().item())
        progress_bar.set_description(
            "Mean so far: {0:.2f}, Last: {1:.2f}, {2} so far: {3:.2f}, Last {4:.2f}".format(
                np.mean(psnrs),
                psnrs[-1],
                dagger_name,
                np.mean(psnrs_dagger),
                psnrs_dagger[-1],
            )
        )
        if i == 0:
            y_out = y
            x_out = x
            recon_out = recon
        if only_first:
            break
    mean_psnr = np.mean(psnrs)
    mean_psnr_dagger = np.mean(psnrs_dagger)
    print("Mean PSNR over the test set: {0:.2f}".format(mean_psnr))
    print(
        "Mean PSNR "
        + dagger_name
        + " over the test set: {0:.2f}".format(mean_psnr_dagger)
    )

    # plot ground truth, observation and reconstruction for the first image from the test dataset
    if plot_example:
        plot([x_out, y_out, recon_out])
    return mean_psnr


best_lmbd = -1
best_psnr = -1000
for lmbd in lmbd_choices:
    mean_psnr = eval_TV(lmbd, validation_dataset, plot_example=False)
    if mean_psnr > best_psnr:
        best_lmbd = lmbd
        best_psnr = mean_psnr
print(
    "Best lambda: {0}, best PSNR on validation set: {1:.2f}".format(
        best_lmbd, best_psnr
    )
)

eval_TV(best_lmbd, dataset, plot_example=True)
