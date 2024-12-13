from deepinv.optim import Prior
from deepinv.models import ICNN
import torch


class ICNNPrior(Prior):
    def __init__(
        self,
        in_channels=3,
        num_filters=64,
        kernel_dim=5,
        num_layers=10,
        strong_convexity=0.5,
        pos_weights=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained=None,
    ):
        super().__init__()
        self.icnn = ICNN(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_dim=kernel_dim,
            num_layers=num_layers,
            strong_convexity=strong_convexity,
            pos_weights=pos_weights,
            device=device,
        )
        self.add_module("ICNN", self.icnn)
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))

    def g(self, x):
        return self.icnn(x)

    def grad(self, x):
        with torch.enable_grad():
            x_ = x.clone()
            x_.requires_grad_(True)
            val = torch.sum(self.g(x_))
            grad = torch.autograd.grad(val, x_, create_graph=True)[0]
        return grad
