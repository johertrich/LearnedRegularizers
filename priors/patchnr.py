"""
Here will be the implementation for the PatchNR 

Altekrueger et al., PatchNR: learning from very few images by patch normalizing flow regularization, Inverse Problems 2023
https://iopscience.iop.org/article/10.1088/1361-6420/acce5e

The PatchNR used GlowCouplingBlocks and random permutations. 
The implementation here is heavily inspired from the FrEIA library (https://github.com/vislearn/FrEIA).

(there actually is a bug in Deepinv at the moment Jan 2024, where no permutation are used)

"""

from deepinv.optim import Prior 
from deepinv.utils import patch_extractor

import torch 
import numpy as np 

class RandomPermutation(torch.nn.Module):
    def __init__(self, dims_in, seed: int = None):
        super().__init__()

        self.dims_in = dims_in

        if seed is not None:
            np.random.seed(seed)

        self.perm = np.random.permutation(self.dims_in)

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = torch.nn.Parameter(torch.LongTensor(self.perm), requires_grad=False)
        self.perm_inv = torch.nn.Parameter(torch.LongTensor(self.perm_inv), requires_grad=False)

    def forward(self, x):
        """
        
        returns: output, log_likelihood
        """

        return x[:, self.perm], 0

    def inverse(self, x):
        """
        
        returns: output, log_likelihood
        """
        return x[:, self.perm_inv], 0


class AffineCouplingBlock(torch.nn.Module):
    def __init__(self, dims_in, subnet_constructor, clamp: float = 2.0, clamp_activation: str = "ATAN"):
        super().__init__()

        self.split_length1 = dims_in // 2 
        self.split_length2 = dims_in - self.split_length1

        self.subnet = subnet_constructor(self.split_length1, self.split_length2 * 2)

        # initialise the last layer of the subnetwork with zero such that the full 
        # block is initialised as the identity 
        if isinstance(self.subnet[-1], torch.nn.Linear):
            self.subnet[-1].weight.data.fill_(0.0)
            self.subnet[-1].bias.data.fill_(0.0)

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = (lambda u: 0.636 * torch.atan(u))
            else:
                raise ValueError(f"Unknown clamp activation {clamp_activation}")

        else:
            self.f_clamp = clamp_activation

        self.clamp = clamp

    def forward(self, x):

        x1, x2 = torch.split(x, [self.split_length1, self.split_length2], dim=1)

        a = self.subnet(x1)
        s, t = a[:, :self.split_length2], a[:, self.split_length2:]
        s = self.clamp * self.f_clamp(s)

        y2 = torch.exp(s) * x2 + t

        log_jac = torch.sum(s, dim=1)

        return torch.cat([x1, y2], dim=1), log_jac

    def inverse(self, x):

        x1, x2 = torch.split(x, [self.split_length1, self.split_length2], dim=1)

        a = self.subnet(x1)
        s, t = a[:, :self.split_length2], a[:, self.split_length2:]
        s = self.clamp * self.f_clamp(s)
        
        y2 = (x2 - t) * torch.exp(-s) 

        log_jac = torch.sum(s, dim=1)

        return torch.cat([x1, y2], dim=1), -log_jac

class INN(torch.nn.Module):
    def __init__(self, dims_in, num_layers=10, sub_net_size=256):
        super().__init__()

        self.dims_in = dims_in
        self.num_layers = num_layers
        self.sub_net_size = sub_net_size

        def subnet_fc(c_in, c_out):
            return torch.nn.Sequential(
                torch.nn.Linear(c_in, self.sub_net_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.sub_net_size, self.sub_net_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.sub_net_size, c_out),
            )

        self.network = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.network.append(AffineCouplingBlock(dims_in=self.dims_in, subnet_constructor=subnet_fc))
            if i < self.num_layers -1:
                self.network.append(RandomPermutation(dims_in=dims_in))

    def forward(self, x):
        log_jac_full = torch.zeros(x.shape[0], device=x.device)

        for module in self.network:
            x, log_jac = module.forward(x)
            log_jac_full += log_jac

        return x, log_jac_full
    
    def inverse(self, x):
        log_jac_full = torch.zeros(x.shape[0], device=x.device)

        for module in self.network[::-1]:
            x, log_jac = module.inverse(x)
            log_jac_full += log_jac

        return x, log_jac_full

class PatchNR(Prior):
    def __init__(self, patch_size=6, n_patches=1000, channels=1,num_layers=10, sub_net_size=256, pad=True, device="cpu"):
        super(PatchNR, self).__init__()

        self.device = device 

        dims_in = patch_size**2 *channels
        self.normalizing_flow = INN(dims_in=dims_in, num_layers=num_layers, sub_net_size=sub_net_size)
        self.normalizing_flow.to(self.device)

        self.n_patches = n_patches
        self.patch_size = patch_size
        self.pad = True 

    def g(self, x, *args, **kwargs):
        r"""
        Evaluates the negative log likelihood function of the PatchNR.

        :param torch.Tensor x: image tensor
        """
        if self.pad:
            x = torch.cat(
                (
                    torch.flip(x[:, :, -self.patch_size : -1, :], (2,)),
                    x,
                    torch.flip(x[:, :, 1 : self.patch_size, :], (2,)),
                ),
                2,
            )
            x = torch.cat(
                (
                    torch.flip(x[:, :, :, -self.patch_size : -1], (3,)),
                    x,
                    torch.flip(x[:, :, :, 1 : self.patch_size], (3,)),
                ),
                3,
            )

        patches, _ = patch_extractor(x, self.n_patches, self.patch_size)

        B, n_patches = patches.shape[0:2]

        # the normalising flow goes from image -> latent 
        latent_x, logdet = self.normalizing_flow(patches.view(B * n_patches, -1)) 
        logpz = 0.5 * torch.sum(latent_x.view(B, n_patches, -1) ** 2, -1)

        nll = logpz - logdet.view(B, n_patches)

        nll = torch.mean(nll, -1)
        return nll


if __name__ == "__main__":

    x = torch.randn(32, 64)
    random_permute = RandomPermutation(dims_in=64)

    z, _ = random_permute.forward(x)
    x_rev, _ = random_permute.inverse(z)

    print(torch.sum((x - x_rev)**2))

    def subnet_fc(c_in, c_out):
        return torch.nn.Sequential(
            torch.nn.Linear(c_in, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, c_out),
        )
    
    affine_block = AffineCouplingBlock(dims_in=64, subnet_constructor=subnet_fc)

    z, log_jac = affine_block.forward(x)
    x_rev, _ = affine_block.inverse(z)
    print(log_jac.shape)
    print(torch.sum((x - x_rev)**2))


    inn = INN(dims_in=64)

    z, log_jac = inn.forward(x)
    x_rev, _ = inn.inverse(z)
    print(log_jac.shape)
    print(torch.sum((x - x_rev)**2))


    print(log_jac)

    x = torch.randn(6, 1, 128, 128)

    patch_nr = PatchNR(patch_size=6)

    nll = patch_nr.g(x)

    print(nll)