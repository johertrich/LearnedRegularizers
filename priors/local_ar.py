"""
Local (patch-based) Adversarial regulariser
"""

from deepinv.optim import Prior
from deepinv.utils import patch_extractor

import torch

import torch.nn as nn
import torch.nn.functional as F


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
        output = self.convnet(image)
        return output

    def grad(self, image):
        with torch.enable_grad():
            image_ = image.clone()
            image_.requires_grad_(True)
            nll = self.g(image_)
            nll = nll.sum()
            grad = torch.autograd.grad(outputs=nll, inputs=image_, create_graph=True)[0]
           
        return grad

class LocalAR(Prior):
    def __init__(
        self,
        patch_size=15,
        n_patches=1000,
        in_channels=1,
        pad=True,
        device="cpu",
        pretrained=None,
        use_bias=True,
        reduction="mean",
        output_factor=1.0,
    ):
        super(LocalAR, self).__init__()

        self.device = device
        self.reduction = reduction
        self.output_factor = output_factor
        self.cnn = cnn(in_channels=in_channels, use_bias=use_bias)
        self.cnn.to(self.device)

        if pretrained is not None:
            try:
                self.cnn.load_state_dict(
                    torch.load(pretrained, map_location=self.device)
                )
            except RuntimeError:
                # sometimes I save the wrong state_dict, i.e., cnn.convnet.0.weight instead of convnet.0.weight
                self.load_state_dict(torch.load(pretrained, map_location=self.device))

        self.n_patches = n_patches
        self.patch_size = patch_size
        self.pad = pad

    def g(self, x, *args, **kwargs):
        r"""
        Evaluates the negative log likelihood function of the PatchNR.

        :param torch.Tensor x: image tensor
        """

        if self.pad:
            pad = self.patch_size - 1
            x = F.pad(x, (pad, pad, pad, pad), mode='replicate')

        if self.n_patches == -1:
            # print("x.shape ", x.shape)
            if self.reduction == "mean":
                out = self.cnn(x).mean([1, 2, 3])
            else:
                out = self.cnn(x).sum([1, 2, 3])
            
        else:
            patches, _ = patch_extractor(x, self.n_patches, self.patch_size)
            B, n_patches = patches.shape[0:2]

            out = self.cnn(
                patches.view(
                    B * n_patches, patches.shape[2], patches.shape[3], patches.shape[3]
                )
            )
            out = out.view(B, n_patches, 1)
            if self.reduction == "mean":
                out = out.mean(1)
            else:
                out = out.sum(1)

        out = out * self.output_factor
        
        return out 

    def grad(self, x, *args, **kwargs):
        r"""
        Evaluates the gradient of the negative log likelihood function of the PatchNR.

        :param torch.Tensor x: image tensor
        """
        with torch.enable_grad():
            x_ = x.clone()
            x_.requires_grad_(True)
            nll = self.g(x_)
            nll = nll.sum()
            grad = torch.autograd.grad(outputs=nll, inputs=x_, create_graph=True)[0]
           
        return grad


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
