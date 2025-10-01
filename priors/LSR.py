import torch
import torch.nn as nn
from deepinv.optim import Prior
from deepinv.models.GSPnP import GSDRUNet


class LSR(Prior):
    def __init__(
        self,
        device="cpu",  # device for the parameters
        pretrained=None,  # if a string is given the weights of the corresponding path are loaded
        nc=[
            32,
            64,
            128,
            256,
        ],  # number of channels at the different stages of the DRUNet (cf the deepinv DRUNet documentation)
        pretrained_denoiser=False,  # set True initialize with the pretrained GSDRUNet weights from deepinv (default False)
        alpha=1.0,  # scaling factor of the DRUNet (cf the deepinv GSDRUNet documentation)
        sigma=0.03,  # noise level input for the DRUNet
    ):
        """
        Defines the Least Squares Residual regularizer defined by R(x)=||x-D(x)||^2, where D is a DRUNet.

        The input arguments are specified as comments above.
        """
        super(LSR, self).__init__()

        self.model = GSDRUNet(
            alpha=alpha,
            in_channels=1,
            out_channels=1,
            nb=2,
            nc=nc,
            act_mode="s",
            pretrained="download" if pretrained_denoiser else None,
            device=device,
        )

        self.model.detach = False

        self.sigma = sigma

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))

    def grad(self, x):
        return self.model.potential_grad(x, self.sigma)

    def g(self, x):
        return self.model.potential(x, self.sigma)
