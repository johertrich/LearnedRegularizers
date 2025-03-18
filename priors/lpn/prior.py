"""Functions to evaluate the learned prior of LPN."""

import numpy as np
import torch
from tqdm import tqdm

from .invert_model import invert


def evaluate_prior(x, model, inv_alg, batch=True):
    """Evaluate the learned prior at x.
    Inputs:
        x: (n, *), numpy.ndarray, n points to evaluate the prior at
        model: the LPN model
        inv_alg: Inversion algorithm, choose from ['ls', 'cvx_cg', 'cvx_gd']
        batch: whether calculate in batch mode

    Outputs:
        dict
        p: (n, ), numpy.ndarray, the prior value at x
        y: (n, *), numpy.ndarray, the inverse of model at x
        fy: (n, *), numpy.ndarray, the model output at y

    Note: The shape of x should match the input shape of model.

    Formula: phi(f(y)) = <y, f(y)> - 1/2 ||f(y)||^2 - psi(y)

    """
    def _batch_run(x, model, inv_alg):
        n = x.shape[0]  # batch size
        device = next(model.parameters()).device
        x_torch = torch.tensor(x).float().to(device)

        # invert model
        y = invert(x, model, inv_alg)
        fy = model(x_torch).detach().cpu().numpy()

        # compute prior
        y_torch = torch.tensor(y).float().to(device)
        psi = model.scalar(y_torch).squeeze(1).detach().cpu().numpy()
        q = 0.5 * np.sum(x.reshape(n, -1) ** 2, axis=1)  # quadratic term
        ip = np.sum(y.reshape(n, -1) * x.reshape(n, -1), axis=1)  # inner product
        p = ip - q - psi

        return p, y, fy

    if batch:
        p, y, fy = _batch_run(x, model, inv_alg)

    else:
        p, y, fy = [], [], []
        for x_i in tqdm(x):
            p_i, y_i, fy_i = _batch_run(x_i[np.newaxis], model, inv_alg)
            p.append(p_i[0])
            y.append(y_i[0])
            fy.append(fy_i[0])
        p = np.array(p)
        y = np.array(y)
        fy = np.array(fy)

    return {"p": p, "y": y, "fy": fy}
