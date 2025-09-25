"""
LearnedPrimalDual Network adapted from dival: https://github.com/jleuschn/dival/blob/master/dival/reconstructors/networks/iterative.py

"""

import torch
import torch.nn as nn
import numpy as np

from .unet import get_unet_model


class IterativeBlock(nn.Module):
    def __init__(
        self,
        n_in=3,
        n_out=1,
        n_memory=5,
        n_layer=3,
        internal_ch=32,
        kernel_size=3,
        batch_norm=True,
        lrelu_coeff=0.2,
    ):
        super(IterativeBlock, self).__init__()
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        modules = []
        if batch_norm:
            modules.append(nn.BatchNorm2d(n_in + n_memory))
        for i in range(n_layer - 1):
            input_ch = (n_in + n_memory) if i == 0 else internal_ch
            modules.append(
                nn.Conv2d(
                    input_ch, internal_ch, kernel_size=kernel_size, padding=padding
                )
            )
            if batch_norm:
                modules.append(nn.BatchNorm2d(internal_ch))

            modules.append(nn.LeakyReLU(lrelu_coeff, inplace=True))
        modules.append(
            nn.Conv2d(
                internal_ch, n_out + n_memory, kernel_size=kernel_size, padding=padding
            )
        )
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        upd = self.block(x)
        return upd


class PrimalDualNet(nn.Module):
    def __init__(
        self,
        image_width,
        num_angles,
        n_iter,
        op,
        op_adj,
        op_init=None,
        n_primal=5,
        n_dual=5,
        use_sigmoid=False,
        n_layer=4,
        internal_ch=32,
        kernel_size=3,
        batch_norm=True,
        lrelu_coeff=0.2,
    ):
        super(PrimalDualNet, self).__init__()
        self.image_width = image_width
        self.num_detector_pixels = int(
            np.ceil(np.sqrt(self.image_width**2 + self.image_width**2))
        )  # this is the default for parallel beam geometry
        self.num_angles = num_angles

        self.n_iter = n_iter

        self.op = op
        self.op_adj = op_adj
        self.op_init = op_init

        self.n_primal = n_primal
        self.n_dual = n_dual
        self.use_sigmoid = use_sigmoid

        self.primal_blocks = nn.ModuleList()
        self.dual_blocks = nn.ModuleList()
        for it in range(n_iter):
            self.dual_blocks.append(
                IterativeBlock(
                    n_in=3,
                    n_out=1,
                    n_memory=self.n_dual - 1,
                    n_layer=n_layer,
                    internal_ch=internal_ch,
                    kernel_size=kernel_size,
                    batch_norm=batch_norm,
                    lrelu_coeff=lrelu_coeff,
                )
            )
            self.primal_blocks.append(
                get_unet_model(
                    in_ch=2 + self.n_primal - 1,
                    out_ch=1 + self.n_primal - 1,
                    scales=4,
                    skip=8,
                    channels=(16, 32, 32, 64),
                    use_sigmoid=False,
                    use_norm=True,
                )
            )

    def forward(self, y):
        primal_cur = torch.zeros(
            y.shape[0],
            self.n_primal,
            self.image_width,
            self.image_width,
            device=y.device,
        )

        if self.op_init is not None:
            primal_cur[:] = self.op_init(y)  # broadcast across dim=1
        dual_cur = torch.zeros(
            y.shape[0],
            self.n_dual,
            self.num_detector_pixels,
            self.num_angles,
            device=y.device,
        )

        for i in range(self.n_iter):
            primal_evalop = self.op(primal_cur[:, 1:2, ...])

            dual_update = torch.cat([dual_cur, primal_evalop, y], dim=1)
            dual_update = self.dual_blocks[i](dual_update)
            dual_cur = dual_cur + dual_update
            dual_evalop = self.op_adj(dual_cur[:, 0:1, ...])

            primal_update = torch.cat([primal_cur, dual_evalop], dim=1)
            primal_update = self.primal_blocks[i](primal_update)
            primal_cur = primal_cur + primal_update
        x = primal_cur[:, 0:1, ...]
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x
