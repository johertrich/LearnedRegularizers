import random
from deepinv.physics import Denoising, GaussianNoise, Tomography
from deepinv.optim import L2
from dataset import get_dataset
import torch
import numpy as np


def get_evaluation_setting(problem, device, root=None):
    physics, data_fidelity = get_operator(problem, device)
    if problem == "Denoising":
        dataset = get_dataset("BSD68")
    elif problem == "CT":
        dataset = get_dataset("LoDoPaB", test=True, root=root)
    return dataset, physics, data_fidelity


def get_operator(problem, device, MAID = False):
    if problem == "Denoising":
        noise_level = 0.1
        if not MAID:
            physics = Denoising(noise_model=GaussianNoise(sigma=noise_level)) 
        else:
            physics = Denoising(noise_model=GaussianNoise_MAID(sigma=noise_level))
        data_fidelity = L2(sigma=1.0)
    elif problem == "CT":
        noise_level = 0.7
        physics = Tomography(
            angles=60,
            img_width=362,
            circle=False,
            device=device,
            noise_model=GaussianNoise(sigma=noise_level),
        )

        def fbp(y):
            out = physics.iradon(y)
            out = out[:, :, 2:-2, 2:-2]
            out = torch.nn.functional.pad(out, (2, 2, 2, 2), mode="replicate")
            return out

        physics.A_dagger = fbp
        data_fidelity = L2(sigma=1.0)
    return physics, data_fidelity


class GaussianNoise_MAID(torch.nn.Module):
    def __init__(self, sigma=0.1, rng: torch.Generator = torch.default_generator):
        super().__init__()
        self.sigma = sigma
        self.rng = rng
        self.noise = None
        self.noise_validation = None
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        np.random.seed(0)
        random.seed(0)
    def forward(self, x):
        """
        Adds Gaussian noise to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Noisy tensor.
        """
        if self.noise == None:
            self.noise = torch.randn_like(x) * self.sigma
            self.noise = self.noise.cpu().detach()
            noise = self.noise.to(x.device)
        elif self.noise.shape != x.shape:
            if self.noise_validation == None:
                self.noise_validation = torch.randn_like(x) * self.sigma
                self.noise_validation = self.noise_validation.cpu().detach()
            noise = self.noise_validation.to(x.device)
        else:
            noise = self.noise.to(x.device)
        return x + noise