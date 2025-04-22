from deepinv.physics import Denoising, GaussianNoise, Tomography
from deepinv.optim import L2
from dataset import get_dataset


def get_evaluation_setting(problem, device):
    if problem == "Denoising":
        noise_level = 0.1
        physics = Denoising(noise_model=GaussianNoise(sigma=noise_level))
        data_fidelity = L2(sigma=1.0)
        dataset = get_dataset("BSD68")
    elif problem == "CT":
        dataset = get_dataset("LoDoPaB", test=True)
        noise_level = 0.1
        physics = Tomography(
            angles=60,
            img_width=dataset[0].shape[-1],
            circle=False,
            device=device,
            noise_model=GaussianNoise(sigma=noise_level),
        )
        data_fidelity = L2(sigma=1.0)

    return dataset, physics, data_fidelity
