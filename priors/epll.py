"""
Here will be the implementation for EPLL 

Zoran et al., From learning models of natural image patches to whole image restoration, IEEE ICCV 2011
https://ieeexplore.ieee.org/abstract/document/6126278

"""


import torch.nn as nn
import torch
import numpy as np
from deepinv.priors.prior import Prior
from deepinv.utils import patch_extractor
from deepinv.optim.utils import conjugate_gradient
from deepinv.models.utils import get_weights_url
from deepinv.optim.utils import GaussianMixtureModel


class EPLL(Prior):
    r"""
    Expected Patch Log Likelihood reconstruction method

    Reconstruction method based on the minimization problem

    .. math::

        \underset{x}{\arg\min} \; \|y-Ax\|^2 - \sum_i \log p(P_ix)

    where the first term is a standard :math:`\ell_2` data-fidelity, and the second term represents a patch prior via
    Gaussian mixture models, where :math:`P_i` is a patch operator that extracts the ith (overlapping) patch from the image.

    #TODO change this
    The reconstruction function is based on the approximated half-quadratic splitting method as in Zoran, D., and Weiss,
    Y.  "From learning models of natural image patches to whole image restoration." (ICCV 2011).

    :param None, deepinv.optim.utils.GaussianMixtureModel GMM: Gaussian mixture defining the distribution on the patch space.
        ``None`` creates a GMM with n_components components of dimension accordingly to the arguments patch_size and channels.
    :param int n_components: number of components of the generated GMM if GMM is ``None``.
    :param str, None pretrained: Path to pretrained weights of the GMM with file ending ``.pt``. None for no pretrained weights,
        ``"download"`` for pretrained weights on the BSDS500 dataset, ``"GMM_lodopab_small"`` for the weights from the limited-angle CT example.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param int patch_size: patch size.
    :param int channels: number of color channels (e.g. 1 for gray-valued images and 3 for RGB images)
    :param str device: defines device (``cpu`` or ``cuda``)
    """

    def __init__(
        self,
        GMM=None,
        n_gmm_components=200,
        patch_size=6,
        channels=1,
        device="cpu",
        pretrained=None
    ):
        super(EPLL, self).__init__()
        if GMM is None:
            self.GMM = GaussianMixtureModel(
                n_gmm_components, patch_size**2 * channels, device=device
            )
        else:
            self.GMM = GMM

        self.patch_size = patch_size
        self.n_components = n_gmm_components
        self.n_channels = channels

        if pretrained is not None: 
            if pretrained[-3:] == ".pt":
                ckpt = torch.load(pretrained)
            else:
                raise ValueError("Pretrained weights have to be provided as a .pt file.")
            self.GMM.load_state_dict(ckpt)
            self.n_components = self.GMM.n_components
            self.patch_size = int(np.sqrt(self.GMM.dimension))

        self.device = device

    def forward(self, y, physics, sigma=None, x_init=None, betas=None, batch_size=-1):
        r"""
        Approximated half-quadratic splitting method for image reconstruction as proposed by Zoran and Weiss.

        :param torch.Tensor y: tensor of observations. Shape: batch size x ...
        :param torch.Tensor, None x_init: tensor of initializations. If ``None`` uses initializes with the adjoint of the forward operator.
            Shape: batch size x channels x height x width
        :param deepinv.physics.LinearPhysics physics: Forward linear operator.
        :param list[float] betas: parameters from the half-quadratic splitting. ``None`` uses the standard choice ``[1,4,8,16,32]/sigma_sq``
        :param int batch_size: batching the patch estimations for large images. No effect on the output, but a small value reduces the memory consumption
            but might increase the computation time. -1 for considering all patches at once.
        """

        x_init = physics.A_adjoint(y) if x_init is None else x_init

        if sigma is None:
            if hasattr(physics.noise_model, "sigma"):
                sigma = physics.noise_model.sigma
            else:
                raise ValueError(
                    "Noise level sigma has to be provided if not present in the physics model."
                )

        if betas is None:
            # default choice as suggested in Parameswaran et al. "Accelerating GMM-Based Patch Priors for Image Restoration: Three Ingredients for a 100× Speed-Up"
            betas = [beta / sigma**2 for beta in [1.0, 4.0, 8.0, 16.0, 32.0]]
        if y.shape[0] > 1:
            # vectorization over a batch of images not implemented....
            return torch.cat(
                [
                    self.reconstruction(
                        y[i : i + 1],
                        x_init[i : i + 1],
                        betas=betas,
                        batch_size=batch_size,
                    )
                    for i in range(y.shape[0])
                ],
                0,
            )
        x = x_init
        Aty = physics.A_adjoint(y)
        for beta in betas:
            x = self._hqs_reconstruction_step(Aty, x, sigma**2, beta, physics, batch_size)
        return x


    def negative_log_likelihood(self, x):
        r"""
        Takes patches and returns the negative log likelihood of the GMM for each patch.

        :param torch.Tensor x: tensor of patches of shape batch_size x number of patches per batch x patch_dimensions
        """
        B, n_patches = x.shape[0:2]
        logpz = self.GMM(x.view(B * n_patches, -1))
        return logpz.view(B, n_patches)


    def _hqs_reconstruction_step(self, Aty, x, sigma_sq, beta, physics, batch_size):
        # precomputations for GMM with covariance regularization
        self.GMM.set_cov_reg(1.0 / beta)
        N, M = x.shape[2:4]
        total_patch_number = (N - self.patch_size + 1) * (M - self.patch_size + 1)

        if batch_size == -1 or batch_size > total_patch_number:
            batch_size = total_patch_number

        # compute sum P_i^T z and sum P_i^T P_i on the fly with batching
        x_tilde_flattened = torch.zeros_like(x).reshape(-1)
        patch_multiplicities = torch.zeros_like(x).reshape(-1)

        # batching loop over all patches in the image
        ind = 0
        while ind < total_patch_number:
            # extract patches
            n_patches = min(batch_size, total_patch_number - ind)
            patch_inds = torch.LongTensor(range(ind, ind + n_patches)).to(x.device)
            patches, linear_inds = patch_extractor(
                x, n_patches, self.patch_size, position_inds_linear=patch_inds
            )
            patches = patches.reshape(patches.shape[0] * patches.shape[1], -1)
            linear_inds = linear_inds.reshape(patches.shape[0], -1)

            # Gaussian selection
            k_star = self.GMM.classify(patches, cov_regularization=True)

            # Patch estimation
            estimation_matrices = torch.bmm(
                self.GMM.get_cov_inv_reg(), self.GMM.get_cov()
            )
            estimation_matrices_k_star = estimation_matrices[k_star]
            patch_estimates = torch.bmm(
                estimation_matrices_k_star, patches[:, :, None]
            ).reshape(patches.shape[0], patches.shape[1])

            # update on-the-fly parameters
            # the following two lines are the same like
            # patch_multiplicities[linear_inds] += 1.0
            # x_tilde_flattened[linear_inds] += patch_estimates
            # where values of multiple indices are accumulated.
            patch_multiplicities.index_put_(
                (linear_inds,), torch.ones_like(patch_estimates), accumulate=True
            )
            x_tilde_flattened.index_put_(
                (linear_inds,), patch_estimates, accumulate=True
            )
            ind = ind + n_patches
        # compute x_tilde
        x_tilde_flattened /= patch_multiplicities

        # Image estimation by CG method
        rhs = Aty + beta * sigma_sq * x_tilde_flattened.view(x.shape)
        op = lambda im: physics.A_adjoint(physics.A(im)) + beta * sigma_sq * im
        hat_x = conjugate_gradient(op, rhs, max_iter=1e2, tol=1e-5)
        return hat_x

    def g(self, x, *args, **kwargs):
        r"""
        Evaluates the negative log likelihood function of the EPLL

        :param torch.Tensor x: image tensor
        """
        
        num_patches = (x.shape[2] - self.patch_size + 1) * (x.shape[3] - self.patch_size + 1)

        patches, _ = patch_extractor(
            x, num_patches, self.patch_size
        )
        if patches.shape[1] != num_patches:
            raise ValueError("Number of patches extracted is not equal to the expected number of patches")
        
        nll = self.GMM.forward(patches.view(patches.shape[0] * num_patches, -1))

        return nll.mean(-1)

    def grad(self, x, *args, **kwargs):
        r"""
        Evaluates the gradient of the negative log likelihood function of the EPLL

        :param torch.Tensor x: image tensor
        """
        with torch.enable_grad():
            x_ = x.clone()
            x_.requires_grad_(True)
            nll = self.g(x).sum()
            grad = torch.autograd.grad(outputs=nll, inputs=x)[0]         
        return grad