"""
Input Convex Neural Networks (ICNN) implementation for learned regularization.

This module implements Input Convex Neural Networks that maintain convexity 
with respect to their input.

Based on https://arxiv.org/abs/1609.07152

"""

import torch
import torch.nn as nn
from deepinv.optim import Prior
import torch.nn.utils.parametrize as P


class ZeroMean(nn.Module):
    """
    Parametrization module that enforces zero mean on convolutional filters.

    This constraint improves performance by ensuring that the convolution
    filters have zero mean across spatial dimensions, which helps with
    training stability and prevents certain bias issues.
    """

    def forward(self, x):
        """
        Apply zero mean constraint to the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape (out_channels, in_channels, height, width)

        Returns:
            torch.Tensor: Zero-mean version of input tensor
        """
        return x - torch.mean(x, dim=(1, 2, 3), keepdim=True)


class ICNN_2l(nn.Module):
    """
    Two-layer Input Convex Neural Network (ICNN).

    This implementation maintains convexity with respect to the input by:
    1. Using non-negative weights for skip connections (wz layer)
    2. Applying increasing convex activation functions
    3. Enforcing structural constraints during training

    The network architecture consists of:
    - Input convolution layer (wx) with zero-mean constraint
    - Hidden convolution layer (wz) with non-negative weights
    - Convex activation function (smoothed ReLU or ELU)
    - Scaling parameter for output normalization

    Note here this implements a 2-layer ICNN for simplicity, but deeper
    architectures can be constructed similarly.

    Args:
        in_c (int): Number of input channels
        channels (int): Number of hidden channels
        kernel_size (int, optional): Convolution kernel size. Defaults to 5.
        smoothing (float, optional): Smoothing parameter for activation. Defaults to 0.01.
        act_name (str, optional): Activation function name ("smoothed_relu" or "elu").
                                 Defaults to "smoothed_relu".
    """

    def __init__(
        self, in_c, channels, kernel_size=5, smoothing=0.01, act_name="smoothed_relu"
    ):
        super(ICNN_2l, self).__init__()
        # Store network parameters
        self.in_c = in_c
        self.channels = channels

        # Learnable smoothing parameter (stored in log space for positivity)
        self.smoothing = nn.Parameter(torch.log(torch.tensor(smoothing)))
        self.padding = kernel_size // 2

        # Input convolution layer (can have negative weights)
        self.wx = nn.Conv2d(
            in_c, channels, kernel_size=kernel_size, padding=self.padding, bias=True
        )

        # Hidden convolution layer (must have non-negative weights for convexity)
        self.wz = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=self.padding, bias=True
        )

        # Learnable scaling parameter (stored in log space)
        self.scaling = nn.Parameter(
            torch.log(torch.tensor(0.001)) * torch.ones(1, channels, 1, 1)
        )

        # Define activation function based on type
        if act_name == "smoothed_relu":
            # Smoothed ReLU: convex piecewise quadratic/linear function
            self.act = (
                lambda x: torch.clip(
                    x, torch.zeros_like(self.smoothing), self.smoothing.exp()
                )
                ** 2
                / (2 * self.smoothing.exp())
                + torch.clip(x, self.smoothing.exp())
                - self.smoothing.exp()
            )
        elif act_name == "elu":
            # Exponential Linear Unit (convex for positive inputs)
            self.act = torch.nn.ELU()
        else:
            raise NameError("Unknown activation!")

        # Apply zero-mean parametrization to input weights
        P.register_parametrization(self.wx, "weight", ZeroMean())

    def forward(self, x):
        """
        Forward pass through the ICNN.

        The forward pass consists of:
        1. Clip weights to maintain non-negativity constraint
        2. Apply input convolution with activation
        3. Apply hidden convolution with activation and scaling
        4. Sum over spatial and channel dimensions to get scalar output per sample

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_c, height, width)

        Returns:
            torch.Tensor: Scalar output per sample, shape (batch_size,)
        """
        # Ensure non-negative weights for convexity
        self.zero_clip_weights()

        # First layer: input convolution + activation
        z1 = self.act(self.wx(x))

        # Second layer: hidden convolution + activation + scaling
        z = self.act(self.wz(z1)) * torch.exp(self.scaling)

        # Sum over all dimensions except batch to get scalar output
        return (torch.sum(z.reshape(z.shape[0], -1), dim=1)).view(x.shape[0])

    def zero_clip_weights(self):
        """
        Clip the weights of the hidden layer to be non-negative.

        This is crucial for maintaining convexity of the ICNN with respect to its input.
        Only the hidden layer weights (wz) need to be non-negative; the input layer
        weights (wx) can be arbitrary.

        Returns:
            ICNN_2l: Self reference for method chaining
        """
        self.wz.weight.data.clamp_(0)
        return self

    def init_weight(self):
        """
        Initialize network weights using appropriate strategies.

        Weight initialization strategy:
        - wx (input layer): Xavier normal initialization with zero-mean constraint
        - wz (hidden layer): Exponential initialization to ensure positive weights

        Returns:
            ICNN_2l: Self reference for method chaining
        """
        # Initialize input layer with Xavier normal, then enforce zero mean
        wx = nn.init.xavier_normal_(self.wx.weight)
        self.wx.weight.data = wx.data - torch.mean(wx.data, (2, 3), True)

        # Initialize hidden layer with exponential distribution for positivity
        self.wz.weight.data.normal_(-10.0, 0.1).exp_()
        return self


class ICNNPrior(Prior):
    """
    ICNN-based prior for variational optimization problems.

    This class wraps the ICNN_2l network to provide a prior that can be used
    in variational reconstruction algorithms. It inherits from deepinv's Prior
    class and provides both the energy function and its gradient.

    The regularization functional value is computed by the ICNN, and its gradient
    can be computed efficiently using automatic differentiation.

    Args:
        in_channels (int): Number of input channels
        channels (int): Number of hidden channels in the ICNN
        device (torch.device): Device to run the computations on
        kernel_size (int, optional): Convolution kernel size. Defaults to 5.
        smoothing (float, optional): Smoothing parameter for activation. Defaults to 0.01.
    """

    def __init__(self, in_channels, channels, device, kernel_size=5, smoothing=0.01):
        super().__init__()
        # Initialize and setup the ICNN on the specified device
        self.icnn = ICNN_2l(in_channels, channels, kernel_size, smoothing).to(device)
        self.icnn.init_weight()

    def g(self, x):
        """
        Compute the regularization functional value.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Energy values for each sample in the batch, shape (batch_size,)
        """
        return self.icnn(x)

    def grad(self, x, get_energy=False):
        """
        Compute the gradient of the regularization functional with respect to the input.

        This method uses automatic differentiation to compute \nabla g(x), which is
        needed for gradient-based optimization algorithms.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            get_energy (bool, optional): If True, also return the energy value.
                                        Defaults to False.

        Returns:
            torch.Tensor: Gradient tensor with same shape as input
            tuple: If get_energy=True, returns (energy, gradient)
        """
        with torch.enable_grad():
            # Clone input and enable gradient computation
            x_ = x.clone()
            x_.requires_grad_(True)

            # Compute energy and its gradient
            z = torch.sum(self.g(x_))
            grad = torch.autograd.grad(z, x_, create_graph=True)[0]

        if get_energy:
            return z, grad
        return grad
