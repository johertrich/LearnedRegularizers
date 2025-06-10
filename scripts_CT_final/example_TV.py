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

problem = "CT"  # Select problem setups, which we consider.
only_first = False  # just evaluate on the first image of the dataset for test purposes

############################################################

# reconstruction hyperparameters, might be problem dependent
if problem == "Denoising":
    lmbd = 6e-2  # regularization parameter
    step_size = 0.2  # step sizes in the PDHG
elif problem == "CT":
    lmbd = 5  # regularization parameter
    step_size = 0.005  # step sizes in the PDHG


#############################################################
############# Problem setup and evaluation ##################
############# This should not be changed   ##################
#############################################################

# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)


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


TVPhysics = LinearPhysics(A=TVDenoiser.nabla, A_adjoint=TVDenoiser.nabla_adjoint)
L1Prior = L1Prior()

physics_list = [physics, TVPhysics]
lambda_list = [1.0, lmbd]
prox_list = [
    lambda u, gamma: data_fidelity.prox_d(u, y, gamma=gamma),
    L1Prior.prox,
]

optimizer = ParallelPrimalDualOptimizer(
    physics_list,
    prox_list,
    lambda_list,
    20000,
    step_size*1e3,
    step_size/1e3,
    stopping_criterion=1e-5,
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
psnr = PSNR(max_pixel=None)

psnrs = []
psnrs_FBP = []
for i, x in (progress_bar := tqdm(enumerate(dataloader))):
    if device == "mps":
        # mps does not support float64
        x = x.to(torch.float32).to(device)
    else:
        x = x.to(device).to(torch.float32)
    y = physics(x)
    x_init = physics.A_dagger(y)
    psnrs_FBP.append(psnr(x_init, x).squeeze().item())
    recon = optimizer.optimize(x_init)
    psnrs.append(psnr(recon, x).squeeze().item())
    progress_bar.set_description(
        "Mean so far: {0:.2f}, Last: {1:.2f}, FBP so far: {2:.2f}, Last {3:.2f}".format(
            np.mean(psnrs), psnrs[-1], np.mean(psnrs_FBP), psnrs_FBP[-1]
        )
    )
    if i == 0:
        y_out = y
        x_out = x
        recon_out = recon
    if only_first:
        break
mean_psnr = np.mean(psnrs)
mean_psnr_FBP = np.mean(psnrs_FBP)
print("Mean PSNR over the test set: {0:.2f}".format(mean_psnr))
print("Mean PSNR FBP over the test set: {0:.2f}".format(mean_psnr_FBP))

# plot ground truth, observation and reconstruction for the first image from the test dataset
plot([x_out, y_out, recon_out])
