from deepinv.physics import Denoising, GaussianNoise, Tomography
from deepinv.optim import L2
from dataset import get_dataset
import torch


def get_evaluation_setting(problem, device, root):
    physics, data_fidelity = get_operator(problem, device)
    if problem == "Denoising":
        dataset = get_dataset("BSD68")
    elif problem == "CT":
        dataset = get_dataset("LoDoPaB", test=True, root=root)
    return dataset, physics, data_fidelity


def get_operator(problem, device):
    if problem == "Denoising":
        noise_level = 0.1
        physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
        data_fidelity = L2(sigma=1.0)
    elif problem == "CT":
        noise_level = 0.7
        physics = Tomography(
            angles=60,
            img_width=362,
            circle=False,
            device=device,
            noise_model=GaussianNoise(sigma=noise_level)
        )

        def fbp(y):
            out = physics.iradon(y)
            out = out[:, :, 2:-2, 2:-2]
            out = torch.nn.functional.pad(out, (2, 2, 2, 2), mode="replicate")
            return out

        physics.A_dagger = fbp
        data_fidelity = L2(sigma=1.0)
    return physics, data_fidelity
