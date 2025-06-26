import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

import math


__all__ = ["Conv2d", "ConvScale2d", "ConvScaleTranspose2d"]


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        invariant=False,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        zero_mean=False,
        bound_norm=False,
        pad=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.invariant = invariant
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.reduction_dim = (1, 2, 3)
        self.pad = pad

        p = kernel_size // 2
        self.pad_op = Pad(pad=p, mode="reflect")

        # add the parameter
        if self.invariant:
            assert self.kernel_size == 3
            weight = torch.empty(out_channels, in_channels, 1, 3)
        else:
            weight = torch.empty(
                out_channels, in_channels, self.kernel_size, self.kernel_size
            )
        self.weight = nn.Parameter(weight)
        # insert them using a normal distribution
        nn.init.normal_(
            self.weight.data, 0.0, math.sqrt(1 / (in_channels * kernel_size**2))
        )

    def get_weight(self):
        if self.invariant:
            weight = torch.empty(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
                device=self.weight.device,
            )
            weight[:, :, 1, 1] = self.weight[:, :, 0, 0]
            weight[:, :, ::2, ::2] = self.weight[:, :, 0, 2].view(
                self.out_channels, self.in_channels, 1, 1
            )
            weight[:, :, 1::2, ::2] = self.weight[:, :, 0, 1].view(
                self.out_channels, self.in_channels, 1, 1
            )
            weight[:, :, ::2, 1::2] = self.weight[:, :, 0, 1].view(
                self.out_channels, self.in_channels, 1, 1
            )
        else:
            weight = self.weight

        if self.zero_mean:
            weight = weight - torch.mean(weight, dim=self.reduction_dim, keepdim=True)
        if self.bound_norm:
            norm = torch.sum(weight**2, dim=self.reduction_dim, keepdim=True).sqrt()
            weight = weight / norm.clip(min=1)

        return weight

    def forward(self, x):
        # construct the kernel
        weight = self.get_weight()
        # compute the convolution
        x = self.pad_op(x)
        return F.conv2d(
            x, weight, self.bias, self.stride, 0, self.dilation, self.groups
        )

    def backward(self, x, output_shape=None):
        # construct the kernel
        weight = self.get_weight()

        # determine the output padding
        if not output_shape is None:
            output_padding = (
                output_shape[2] - ((x.shape[2] - 1) * self.stride + 1),
                output_shape[3] - ((x.shape[3] - 1) * self.stride + 1),
            )
        else:
            output_padding = 0

        # compute the convolution
        x = F.conv_transpose2d(
            x,
            weight,
            self.bias,
            self.stride,
            0,
            output_padding,
            self.groups,
            self.dilation,
        )
        x = self.pad_op.backward(x)
        return x

    def extra_repr(self):
        s = "({out_channels}, {in_channels}, {kernel_size}), invariant={invariant}"
        if self.stride != 1:
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.bias is None:
            s += ", bias=True"
        if self.zero_mean:
            s += ", zero_mean={zero_mean}"
        if self.bound_norm:
            s += ", bound_norm={bound_norm}"
        return s.format(**self.__dict__)


class ConvScale2d(Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        invariant=False,
        groups=1,
        stride=2,
        bias=False,
        zero_mean=False,
        bound_norm=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            invariant=invariant,
            stride=stride,
            dilation=1,
            groups=groups,
            bias=bias,
            zero_mean=zero_mean,
            bound_norm=bound_norm,
        )

        # create the convolution kernel
        if self.stride > 1:
            k = torch.as_tensor([1, 4, 6, 4, 1], dtype=torch.float32)[:, None]
            k = k @ k.T
            k /= k.sum()
            self.register_buffer("blur", k.reshape(1, 1, 5, 5))

        # overwrite padding op
        p = (kernel_size + 4 * stride // 2) // 2
        self.pad_op = Pad(pad=p, mode="reflect")

    def get_weight(self):
        weight = super().get_weight()
        if self.stride > 1:
            weight = weight.reshape(-1, 1, self.kernel_size, self.kernel_size)
            for i in range(self.stride // 2):
                weight = F.conv2d(weight, self.blur, padding=4)
            weight = weight.reshape(
                self.out_channels,
                self.in_channels,
                self.kernel_size + 2 * self.stride,
                self.kernel_size + 2 * self.stride,
            )
        return weight


class ConvScaleTranspose2d(ConvScale2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        invariant=False,
        groups=1,
        stride=2,
        bias=False,
        zero_mean=False,
        bound_norm=False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            invariant=invariant,
            groups=groups,
            stride=stride,
            bias=bias,
            zero_mean=zero_mean,
            bound_norm=bound_norm,
        )

    def forward(self, x, output_shape):
        return super().backward(x, output_shape)

    def backward(self, x):
        return super().forward(x)


class Pad(torch.nn.Module):
    def __init__(self, pad, mode="reflect", value=0):
        super().__init__()
        self.pad = pad
        pad = (pad, pad, pad, pad)
        self.f = lambda x: torch.nn.functional.pad(
            x, pad=pad, mode="constant", value=value
        )

    def forward(self, x):
        if self.pad > 0:
            x = self.f(x)
        return x

    def backward(self, x):
        if self.pad > 0:
            p = self.pad
            return x[..., p:-p, p:-p]
        else:
            return x


# class Pad(torch.nn.Module):
#     def __init__(self, pad, mode="reflect", value=0):
#         super().__init__()
#         self.pad = pad
#         pad = (pad, pad, pad, pad)
#         self.f = lambda x: torch.nn.functional.pad(
#             x, pad=pad, mode=mode, value=value
#         )

#     def forward(self, x):
#         if self.pad > 0:
#             x = self.f(x)
#         return x

#     def backward(self, x):
#         if self.pad > 0:
#             n, c, h, w = x.shape
#             with torch.enable_grad():
#                 tmp = x.new_ones(n, c, h-2*self.pad, w-2*self.pad).requires_grad_(True)
#                 out = torch.autograd.grad(self.f(tmp), tmp, grad_outputs=x, create_graph=x.requires_grad)[0]
#             if not x.requires_grad:
#                 out = out.detach()
#             return out
#         else:
#             return x
