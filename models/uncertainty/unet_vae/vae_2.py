"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
from models.unet.unet_model import UnetModel


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class Autoencoder(nn.Module):
    """
    Conditional autoencoder. Models P(X | hat{X}) where hat{X} is the zero filled reconstruction and
    X models the distribution over correct reconstruction.

    The decoder architecture is based on a U-Net with an additional sampled latent code as input.

    Concerns:
        - Model might focus soley on the conditional and ignore the sampled latent code.
    """

    def __init__(self, encoding_dim, chans, num_pool_layers, drop_prob):
        self.encoder = Encoder(
            in_chans=2,  # zero-filled solution + target
            encoding_dim=encoding_dim,
            chans=32,
            num_pool_layers=3,
            drop_prob=0.0
        )
        self.decoder = UNet(
            in_chans=(1 + encoding_dim),
            out_chans=1,
            chans=32,
            num_pool_layers=4,
            drop_prob=0.0
        )

    def forward(self, reconstruction, target):
        """
        Args:
            input (torch.Tensor):           Shape: [batch_size, 2, height, width]
        """
        pass


class Encoder(nn.Module):
    """
    """

    def __init__(self, in_chans, encoding_dim, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            encoding_dim (int): Dimension of the encoding space.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.encoding_dim = encoding_dim
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)
        self.mu = nn.Linear(10, encoding_dim)
        self.sigma = nn.Linear(10, encoding_dim)

    def forward(self, input, encoding):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            output = F.max_pool2d(output, kernel_size=2)
        output = self.conv(output)
        from IPython import embed
        embed()

        return output
