import torch
import torch.nn as nn
from deepinv.optim import Prior
import torch.nn.utils.parametrize as P
from .ICNN import ICNN_2l


class IDCNNPrior(Prior):
    def __init__(
        self,
        in_channels,
        channels,
        device,
        kernel_size=5,
        smoothing=0.01,
        act_name="smoothed_relu",
        pretrained=None,
    ):
        super().__init__()
        self.icnn1 = ICNN_2l(
            in_channels, channels, kernel_size, smoothing, act_name=act_name
        ).to(device)
        self.icnn1.init_weight()
        self.icnn2 = ICNN_2l(
            in_channels, channels, kernel_size, smoothing, act_name=act_name
        ).to(device)
        self.icnn2.init_weight()

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))

    def g(self, x):
        return self.icnn1(x) - self.icnn2(x)

    def grad(self, x, get_energy=False):
        with torch.enable_grad():
            x_ = x.clone()
            x_.requires_grad_(True)
            z = torch.sum(self.g(x_))
            grad = torch.autograd.grad(z, x_, create_graph=True)[0]
        if get_energy:
            return z, grad
        return grad
