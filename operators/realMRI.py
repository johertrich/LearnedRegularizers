from deepinv.physics import LinearPhysics, MRI
import torch


class MRIonR(LinearPhysics):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mri = MRI(*args, **kwargs)

    def A(self, x, **kwargs):
        x = torch.cat((x, torch.zeros_like(x)), 1)
        return self.mri.A(x)

    def A_adjoint(self, y, **kwargs):
        return self.mri.A_adjoint(y)[:, :1, :, :]
