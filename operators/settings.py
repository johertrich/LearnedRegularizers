"""
Here, we define the forward operators, noise levels and data-fidelity terms used in the experiments of the chapter.
We provide two functions:
For training, we use the get_operator function which returns the physics and data fidelity term.
For evaluation, we use the get_evaluation_setting function which additionally returns the test dataset.
"""

from deepinv.physics import Denoising, GaussianNoise, Tomography
from deepinv.optim import L2
from dataset import get_dataset
import torch


def get_evaluation_setting(problem, device, root=None):
    """
    Defines dataset, physics (i.e. forward operator and noise model) and data fidelity term for the evaluation
    of the learned regularizers.

    Input arguments:
    problem - defines which experiment should be evaluated. Use "Denoising" for Experiment 1 of the chapter and "CT" for Experiment 2 and 3.
    device - used device ("cpu" or "cuda")
    root - optional argument to define the root directory of the dataset (by default None, which means that the
        dataset will be loaded from "." if it exists, otherwise it will be downloaded to ".")
    """
    physics, data_fidelity = get_operator(problem, device)
    if problem == "Denoising":
        dataset = get_dataset("BSD68")
    elif problem == "CT":
        dataset = get_dataset("LoDoPaB", test=True, root=root)
    return dataset, physics, data_fidelity


def get_operator(problem, device):
    """
    Defines physics (i.e. forward operator and noise model) and data fidelity term for the evaluation
    of the learned regularizers.

    Input arguments:
    problem - defines which experiment should be evaluated. Use "Denoising" for Experiment 1 of the chapter and "CT" for Experiment 2 and 3.
    device - used device ("cpu" or "cuda")
    """

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
            noise_model=GaussianNoise(sigma=noise_level),
        )

        # small trick to remove boundary artifacts from the FBP
        def fbp(y):
            out = physics.iradon(y)
            out = out[:, :, 2:-2, 2:-2]
            out = torch.nn.functional.pad(out, (2, 2, 2, 2), mode="replicate")
            return out

        physics.A_dagger = fbp
        data_fidelity = L2(sigma=1.0)
    return physics, data_fidelity
