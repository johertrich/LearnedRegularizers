"""
Here will be the implementation for the PatchNR 

Altekrueger et al., PatchNR: learning from very few images by patch normalizing flow regularization, Inverse Problems 2023
https://iopscience.iop.org/article/10.1088/1361-6420/acce5e

The PatchNR uses two-sided AffineCouplingBlocks. 

The implementation here is inspired from the FrEIA library (https://github.com/vislearn/FrEIA).
"""

from deepinv.optim import Prior 
from deepinv.utils import patch_extractor

import torch 

class AffineCouplingBlock(torch.nn.Module):
    """The inputs are split in two halves. Two affine
    coupling operations are performed in turn on both halves of the input."""
    def __init__(self, dims_in, subnet_constructor, clamp: float = 2.0, clamp_activation: str = "ATAN"):
        super().__init__()

        self.split_length1 = dims_in // 2 
        self.split_length2 = dims_in - self.split_length1

        self.subnet1 = subnet_constructor(self.split_length1, self.split_length2 * 2)
        self.subnet2 = subnet_constructor(self.split_length2, self.split_length1 * 2)

        # initialise the last layer of the subnetwork with zero such that the full 
        # block is initialised as the identity 
        if isinstance(self.subnet1[-1], torch.nn.Linear):
            self.subnet1[-1].weight.data.fill_(0.0)
            self.subnet1[-1].bias.data.fill_(0.0)

        if isinstance(self.subnet2[-1], torch.nn.Linear):
            self.subnet2[-1].weight.data.fill_(0.0)
            self.subnet2[-1].bias.data.fill_(0.0)

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = (lambda u: 0.636 * torch.atan(u))
            else:
                raise ValueError(f"Unknown clamp activation {clamp_activation}")

        else:
            self.f_clamp = clamp_activation

        self.clamp = clamp

    def forward(self, x, rev=False):
        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations

        x1, x2 = torch.split(x, [self.split_length1, self.split_length2], dim=1)

        if not rev:
            y1, j1 = self._coupling1(x1, x2)

            y2, j2 = self._coupling2(x2, y1)
        else:
            # names of x and y are swapped for the reverse computation
            y2, j2 = self._coupling2(x2, x1, rev=True)

            y1, j1 = self._coupling1(x1, y2, rev=True)

        return torch.cat((y1, y2), 1), j1 + j2


    def _coupling1(self, x1, u2, rev=False):
        # notation (same for _coupling2):
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        a2 = self.subnet2(u2)
        s2, t2 = a2[:, :self.split_length1], a2[:, self.split_length1:]
        s2 = self.clamp * self.f_clamp(s2)

        j1 = torch.sum(s2, dim=1)

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            return y1, -j1
        else:
            y1 = torch.exp(s2) * x1 + t2
            return y1, j1

    def _coupling2(self, x2, u1, rev=False):
        a1 = self.subnet1(u1)
        s1, t1 = a1[:, :self.split_length2], a1[:, self.split_length2:]
        s1 = self.clamp * self.f_clamp(s1)
        j2 = torch.sum(s1, dim=1)

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            return y2, -j2
        else:
            y2 = torch.exp(s1) * x2 + t1
            return y2, j2



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
            
    def forward(self, x, rev=False):
        log_jac_full = torch.zeros(x.shape[0], device=x.device)

        if not rev:
            for module in self.network:
                x, log_jac = module(x, rev=rev)
                log_jac_full += log_jac

            return x, log_jac_full
        else:
            for module in self.network[::-1]:
                x, log_jac = module(x, rev=rev)
                log_jac_full += log_jac

            return x, log_jac_full



class PatchNR(Prior):
    def __init__(self, patch_size=6, n_patches=1000, channels=1,num_layers=10, sub_net_size=256, pad=True, device="cpu", pretrained=None):
        super(PatchNR, self).__init__()

        self.device = device 

        dims_in = patch_size**2 *channels
        self.normalizing_flow = INN(dims_in=dims_in, num_layers=num_layers, sub_net_size=sub_net_size)
        self.normalizing_flow.to(self.device)

        if pretrained is not None:
            self.normalizing_flow.load_state_dict(torch.load(pretrained, map_location=self.device))

        self.n_patches = n_patches
        self.patch_size = patch_size
        self.pad = pad 

    def g(self, x, *args, **kwargs):
        r"""
        Evaluates the negative log likelihood function of the PatchNR.

        :param torch.Tensor x: image tensor
        """
        return_patch_per_pixel = kwargs.get('return_patch_per_pixel', False)

        if self.pad:
            x = torch.cat(
                (
                    torch.flip(x[:, :, -self.patch_size : -1, :].detach(), (2,)),
                    x,
                    torch.flip(x[:, :, 1 : self.patch_size, :].detach(), (2,)),
                ),
                2,
            )
            x = torch.cat(
                (
                    torch.flip(x[:, :, :, -self.patch_size : -1].detach(), (3,)),
                    x,
                    torch.flip(x[:, :, :, 1 : self.patch_size].detach(), (3,)),
                ),
                3,
            )

        patches, linear_inds = patch_extractor(x, self.n_patches, self.patch_size)

        B, n_patches = patches.shape[0:2]

        # the normalising flow goes from image -> latent 
        latent_x, logdet = self.normalizing_flow(patches.view(B * n_patches, -1)) 

        logpz = 0.5 * torch.sum(latent_x.view(B, n_patches, -1) ** 2, -1)

        nll = logpz - logdet.view(B, n_patches)

        nll = torch.mean(nll, -1)
        
        if return_patch_per_pixel:
            patch_per_pixel = torch.zeros_like(x).reshape(-1)
            patch_per_pixel.index_put_(
                    (linear_inds,), torch.ones_like(patches).view(-1), accumulate=True
                )
            patch_per_pixel = patch_per_pixel.reshape(x.shape)

            return nll, patch_per_pixel
        else:
            return nll

    def grad(self, x, *args, **kwargs):
        r"""
        Evaluates the gradient of the negative log likelihood function of the PatchNR.

        :param torch.Tensor x: image tensor
        """
        with torch.enable_grad():
            x.requires_grad_()

            nll, patch_per_pixel = self.g(x, return_patch_per_pixel=True)
            nll = nll.sum()
            grad = torch.autograd.grad(outputs=nll, inputs=x)[0] 
            #print(torch.sum(grad**2))
            grad_norm = (patch_per_pixel + 1) / torch.max(patch_per_pixel)
            if self.pad:
                grad_norm = grad_norm[:, :, self.patch_size - 1 : -(self.patch_size - 1), :]
                # Remove the columns added for horizontal padding
                grad_norm = grad_norm[:, :, :, self.patch_size - 1 : -(self.patch_size - 1)]
        #import matplotlib.pyplot as plt 
        #fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
        #ax1.imshow(x[0,0].detach().cpu().numpy())
        #ax2.imshow(grad[0,0].detach().cpu().numpy())
        #ax3.imshow(patch_per_pixel[0,0].detach().cpu().numpy())
        #ax4.imshow((grad /grad_norm)[0,0].detach().cpu().numpy())
        #ax5.imshow(grad_norm[0,0].detach().cpu().numpy())

        #plt.show()

        return grad / grad_norm 

if __name__ == "__main__":

    x = torch.randn(32, 64)
    
    def subnet_fc(c_in, c_out):
        return torch.nn.Sequential(
            torch.nn.Linear(c_in, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, c_out),
        )
    
    affine_block = AffineCouplingBlock(dims_in=64, subnet_constructor=subnet_fc)

    z, log_jac = affine_block(x)
    x_rev, _ = affine_block(z, rev=True)
    print(log_jac.shape)
    print(torch.sum((x - x_rev)**2))


    inn = INN(dims_in=64)

    z, log_jac = inn(x)
    x_rev, _ = inn(z, rev=True)
    print(log_jac.shape)
    print(torch.sum((x - x_rev)**2))


    print(log_jac)

    x = torch.randn(6, 1, 128, 128)

    patch_nr = PatchNR(patch_size=6)

    nll = patch_nr.g(x)

    print(nll)

    reg_grad = patch_nr.grad(x)

    print(reg_grad)