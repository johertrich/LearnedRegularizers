import inspect
import torch
from deepinv.optim import Prior


class ParameterLearningWrapper(Prior):
    def __init__(
        self,
        regularizer,  # regularizer which will be equiped with learnable regularization parameter and scaling
        scale_init=0.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        The parameter class adds a learnable regularization parameter and a learnable scaling factor to
        a predefined regularizer.
        More precisely, it defines a regularizer

        R_tilde(x)=exp(alpha)/exp(s)**2 R(exp(s) x)

        for a predefined base regularizer R, where alpha and s are learnable parameters. The exponential function
        enforces these scaling parameters to be positive.

        Parameters of the Constructor:
        regularizer - base regularizer R which is used
        scale_init - initialization of the parameter s (optional, default 0)
        device - device of the paramterers (optional, default uses cuda if available and cpu otherwise)
        """
        super().__init__()
        self.regularizer = regularizer
        self.add_module("regularizer", self.regularizer)
        self.alpha = torch.nn.Parameter(
            torch.tensor(0.0, device=device, requires_grad=True)
        )
        self.scale = torch.nn.Parameter(
            torch.tensor(scale_init, device=device, requires_grad=True)
        )
        signature = inspect.signature(self.regularizer.grad)
        argument_names = [param.name for param in signature.parameters.values()]
        self.has_get_energy = False
        if "get_energy" in argument_names:
            self.has_get_energy = True

    def g(self, x):
        return torch.exp(self.alpha - 2 * self.scale) * self.regularizer.g(
            torch.exp(self.scale) * x
        )

    def grad(self, x, get_energy=False):
        if not self.has_get_energy:
            if get_energy:
                return torch.exp(self.alpha - 2 * self.scale) * self.regularizer.g(
                    torch.exp(self.scale) * x
                ), torch.exp(self.alpha - self.scale) * self.regularizer.grad(
                    torch.exp(self.scale) * x
                )
            else:
                return torch.exp(self.alpha - self.scale) * self.regularizer.grad(
                    torch.exp(self.scale) * x
                )
        reg_out = self.regularizer.grad(
            torch.exp(self.scale) * x, get_energy=get_energy
        )
        if get_energy:
            return (
                torch.exp(self.alpha - 2 * self.scale) * reg_out[0],
                torch.exp(self.alpha - self.scale) * reg_out[1],
            )
        return torch.exp(self.alpha - self.scale) * reg_out
