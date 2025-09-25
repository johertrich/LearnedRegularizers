import torch
import torch.nn as nn
import torch.nn.functional as F
from deepinv.optim import Prior
import math

from .conv import *

import unittest

__all__ = ["TDV"]


class TDV(Prior):
    """
    total deep variation (TDV) regularizer
    """

    def __init__(
        self,
        in_channels=1,
        num_features=32,
        multiplier=1,
        num_mb=3,
        zero_mean=True,
        num_scales=3,
    ):
        super().__init__()

        self._fn = self.energy

        self.in_channels = in_channels
        self.num_features = num_features
        self.multiplier = multiplier
        self.num_mb = num_mb
        self.pot, self.act = lambda x: x**2 / 2, lambda x: x
        self.zero_mean = zero_mean
        self.num_scales = num_scales

        # construct the regularizer
        self.K1 = Conv2d(
            self.in_channels,
            self.num_features,
            3,
            zero_mean=self.zero_mean,
            invariant=False,
            bound_norm=True,
            bias=False,
        )
        self.mb = nn.ModuleList(
            [
                MacroBlock(
                    self.num_features,
                    num_scales=self.num_scales,
                    bound_norm=False,
                    invariant=False,
                    multiplier=self.multiplier,
                )
                for _ in range(self.num_mb)
            ]
        )
        self.KN = Conv2d(
            self.num_features, 1, 1, invariant=False, bound_norm=False, bias=False
        )

    def energy(self, x):
        x = self._transformation(x)
        return self._potential(x)

    def g(self, x):
        return self.energy(x).sum(dim=(1, 2, 3))

    def grad(self, x, get_energy=False):
        x = self._transformation(x)
        if get_energy:
            energy = self._potential(x).sum(dim=(1, 2, 3))
        # and its gradient
        x = self._activation(x)
        grad = self._transformation_T(x)
        if get_energy:
            return energy, grad
        else:
            return grad

    def _potential(self, x):
        return self.pot(x) / self.num_features

    def _activation(self, x):
        return self.act(x) / self.num_features

    def _transformation(self, x):
        # extract features
        x = self.K1(x)
        # apply mb
        x = [
            x,
        ] + [None for i in range(self.num_scales - 1)]
        for i in range(self.num_mb):
            x = self.mb[i](x)
        # compute the output
        out = self.KN(x[0])
        return out

    def _transformation_T(self, grad_out):
        # compute the output
        grad_x = self.KN.backward(grad_out)
        # apply mb
        grad_x = [
            grad_x,
        ] + [None for i in range(self.num_scales - 1)]
        for i in range(self.num_mb)[::-1]:
            grad_x = self.mb[i].backward(grad_x)
        # extract features
        grad_x = self.K1.backward(grad_x[0])
        return grad_x


class Softplus2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return Softplus_fun2.apply(x)


@torch.jit.script
def softplusg2(x):
    return torch.exp(x - 2 * F.softplus(x))


class Softplus_fun2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.softplus(x) - math.log(2), F.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_in1, grad_in2):
        x = ctx.saved_tensors[0]
        return F.sigmoid(x) * grad_in1 + softplusg2(x) * grad_in2


class MicroBlock(nn.Module):
    def __init__(self, num_features, bound_norm=False, invariant=False):
        super(MicroBlock, self).__init__()

        self.conv1 = Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            invariant=invariant,
            bound_norm=bound_norm,
            bias=False,
        )
        self.act = Softplus2()
        self.conv2 = Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            invariant=invariant,
            bound_norm=bound_norm,
            bias=False,
        )

        # save the gradient of the the activation function for the backward path
        self.act_prime = None

    def forward(self, x):
        a, ap = self.act(self.conv1(x))
        self.act_prime = ap
        x = x + self.conv2(a)
        return x

    def backward(self, grad_out):
        assert not self.act_prime is None
        out = grad_out + self.conv1.backward(
            self.act_prime * self.conv2.backward(grad_out)
        )
        if not self.act_prime.requires_grad:
            self.act_prime = None
        return out


class MacroBlock(nn.Module):
    def __init__(
        self,
        num_features,
        num_scales=3,
        multiplier=1,
        bound_norm=False,
        invariant=False,
    ):
        super().__init__()

        self.num_scales = num_scales

        # micro blocks
        self.mb = []
        for i in range(num_scales - 1):
            b = nn.ModuleList(
                [
                    MicroBlock(
                        num_features * multiplier**i,
                        bound_norm=bound_norm,
                        invariant=invariant,
                    ),
                    MicroBlock(
                        num_features * multiplier**i,
                        bound_norm=bound_norm,
                        invariant=invariant,
                    ),
                ]
            )
            self.mb.append(b)
        # the coarsest scale has only one microblock
        self.mb.append(
            nn.ModuleList(
                [
                    MicroBlock(
                        num_features * multiplier ** (num_scales - 1),
                        bound_norm=bound_norm,
                        invariant=invariant,
                    )
                ]
            )
        )
        self.mb = nn.ModuleList(self.mb)

        # down/up sample
        self.conv_down = []
        self.conv_up = []
        for i in range(1, num_scales):
            self.conv_down.append(
                ConvScale2d(
                    num_features * multiplier ** (i - 1),
                    num_features * multiplier**i,
                    kernel_size=3,
                    bias=False,
                    invariant=invariant,
                    bound_norm=bound_norm,
                )
            )
            self.conv_up.append(
                ConvScaleTranspose2d(
                    num_features * multiplier ** (i - 1),
                    num_features * multiplier**i,
                    kernel_size=3,
                    bias=False,
                    invariant=invariant,
                    bound_norm=bound_norm,
                )
            )
        self.conv_down = nn.ModuleList(self.conv_down)
        self.conv_up = nn.ModuleList(self.conv_up)

    def forward(self, x):
        assert len(x) == self.num_scales

        # down scale and feature extraction
        for i in range(self.num_scales - 1):
            # 1st micro block of scale
            x[i] = self.mb[i][0](x[i])
            # down sample for the next scale
            x_i_down = self.conv_down[i](x[i])
            if x[i + 1] is None:
                x[i + 1] = x_i_down
            else:
                x[i + 1] = x[i + 1] + x_i_down

        # on the coarsest scale we only have one micro block
        x[self.num_scales - 1] = self.mb[self.num_scales - 1][0](x[self.num_scales - 1])

        # up scale the features
        for i in range(self.num_scales - 1)[::-1]:
            # first upsample the next coarsest scale
            x_ip1_up = self.conv_up[i](x[i + 1], x[i].shape)
            # skip connection
            x[i] = x[i] + x_ip1_up
            # 2nd micro block of scale
            x[i] = self.mb[i][1](x[i])

        return x

    def backward(self, grad_x):

        # backward of up scale the features
        for i in range(self.num_scales - 1):
            # 2nd micro block of scale
            grad_x[i] = self.mb[i][1].backward(grad_x[i])
            # first upsample the next coarsest scale
            grad_x_ip1_up = self.conv_up[i].backward(grad_x[i])
            # skip connection
            if grad_x[i + 1] is None:
                grad_x[i + 1] = grad_x_ip1_up
            else:
                grad_x[i + 1] = grad_x[i + 1] + grad_x_ip1_up

        # on the coarsest scale we only have one micro block
        grad_x[self.num_scales - 1] = self.mb[self.num_scales - 1][0].backward(
            grad_x[self.num_scales - 1]
        )

        # down scale and feature extraction
        for i in range(self.num_scales - 1)[::-1]:
            # down sample for the next scale
            grad_x_i_down = self.conv_down[i].backward(grad_x[i + 1], grad_x[i].shape)
            grad_x[i] = grad_x[i] + grad_x_i_down
            # 1st micro block of scale
            grad_x[i] = self.mb[i][0].backward(grad_x[i])

        return grad_x


# to run execute: python -m unittest [-v] ddr.tdv
class GradientTest(unittest.TestCase):

    def test_tdv_gradient(self):
        # setup the data
        x = torch.rand((2, 1, 64, 64), dtype=torch.float64)

        # define the TDV regularizer
        config = {
            "in_channels": 1,
            # 'out_channels': 1,
            "num_features": 4,
            "num_scales": 3,
            "num_mb": 2,
            "multiplier": 2,
        }
        R = TDV(**config).double()

        def compute_loss(scale):
            return torch.sum(R(scale * x))

        scale = 1.0

        # compute the gradient using the implementation
        grad_scale = torch.sum(x * R.grad(scale * x))

        # check it numerically
        epsilon = 1e-4
        with torch.no_grad():
            l_p = compute_loss(scale + epsilon)
            l_n = compute_loss(scale - epsilon)
            grad_scale_num = (l_p - l_n) / (2 * epsilon)

        condition = (grad_scale - grad_scale_num).abs() < 1e-3
        print(
            f"grad_scale: {grad_scale:.7f} num_grad_scale {grad_scale_num:.7f} success: {condition}"
        )
        self.assertTrue(condition)


if __name__ == "__main__":
    unittest.test()
