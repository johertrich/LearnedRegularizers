"""
Local (patch-based) Adversarial regulariser 
"""

from deepinv.optim import Prior
from deepinv.utils import patch_extractor

import torch

import torch.nn as nn


class cnn(nn.Module):
    def __init__(self, in_channels=1, use_bias=True):
        super(cnn, self).__init__()

        self.act = nn.SiLU()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=0, stride=1),
            self.act,
            nn.Conv2d(128, 64, 3, padding=0, stride=1),
            self.act,
            nn.Conv2d(64, 64, 3, padding=0, stride=1),
            self.act,
            nn.Conv2d(64, 64, 3, padding=0, stride=1),
            self.act,
            nn.Conv2d(64, 32, 3, padding=0, stride=1),
            self.act,
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            self.act,
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=0, bias=use_bias),
        )


    def forward(self, image):
        output = self.convnet(image)  # .squeeze().unsqueeze(-1)
        return output

    def g(self, image):
        output = self.convnet(image).squeeze().unsqueeze(-1)
        return output


class LocalAR(Prior):
    def __init__(
        self,
        patch_size=15,
        n_patches=1000,
        in_channels=1,
        pad=True,
        device="cpu",
        pretrained=None,
        use_bias=True
    ):
        super(LocalAR, self).__init__()

        self.device = device

        self.cnn = cnn(in_channels=in_channels, use_bias=use_bias)
        self.cnn.to(self.device)

        if pretrained is not None:
            self.cnn.load_state_dict(torch.load(pretrained, map_location=self.device))

        self.n_patches = n_patches
        self.patch_size = patch_size
        self.pad = pad

    def g(self, x, *args, **kwargs):
        r"""
        Evaluates the negative log likelihood function of the PatchNR.

        :param torch.Tensor x: image tensor
        """
        return_patch_per_pixel = kwargs.get("return_patch_per_pixel", False)
        # print("RETURN EXTRA: ", return_patch_per_pixel)

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

        if self.n_patches == -1:
            #print("x.shape ", x.shape)
            out = self.cnn(x).mean([1, 2, 3])
            # .mean([1,2,3]).unsqueeze(-1)
            patches, linear_inds = patch_extractor(x, self.n_patches, self.patch_size)
            #print(patches.shape, linear_inds.shape, out.shape)
        else:
            patches, linear_inds = patch_extractor(x, self.n_patches, self.patch_size)
            B, n_patches = patches.shape[0:2]

            out = self.cnn(
                patches.view(
                    B * n_patches, patches.shape[2], patches.shape[3], patches.shape[3]
                )
            )
            out = out.view(B, n_patches, 1)
            out = out.mean(1)

        if return_patch_per_pixel:
            patch_per_pixel = torch.zeros(*x.shape[1:], device=x.device).reshape(-1)
            patch_per_pixel.index_put_(
                (linear_inds,), torch.ones_like(patches[0]).view(-1), accumulate=True
            )
            patch_per_pixel = patch_per_pixel.reshape(x[0].shape)

            return out, patch_per_pixel.unsqueeze(0)
        else:
            return out#.squeeze()

    def grad(self, x, *args, **kwargs):
        r"""
        Evaluates the gradient of the negative log likelihood function of the PatchNR.

        :param torch.Tensor x: image tensor
        """
        with torch.enable_grad():
            x.requires_grad_()

            nll, patch_per_pixel = self.g(x, return_patch_per_pixel=True)
            # print(nll)
            nll = nll.sum()
            grad = torch.autograd.grad(outputs=nll, inputs=x, create_graph=True)[0]
            # print(torch.sum(grad**2))
            grad_norm = (patch_per_pixel + 1) / torch.max(patch_per_pixel)
            if self.pad:
                grad_norm = grad_norm[
                    :, :, self.patch_size - 1 : -(self.patch_size - 1), :
                ]
                # Remove the columns added for horizontal padding
                grad_norm = grad_norm[
                    :, :, :, self.patch_size - 1 : -(self.patch_size - 1)
                ]

        return grad / grad_norm


if __name__ == "__main__":

    x = torch.randn(8, 1, 16, 16)

    prior = LocalAR(n_patches=10, pad=False)

    out_cnn = prior.cnn(x).mean([1, 2, 3])
    print(out_cnn.shape)
    print(out_cnn)

    out = prior.g(x)

    print(out)

    """
    with torch.no_grad():
        out = prior.g(x)

        print(x.shape, out.shape)

        print(out)

        prior.n_patches = -1 
        out = prior.g(x)

        print(x.shape, out.shape)

        print(out)

    """
