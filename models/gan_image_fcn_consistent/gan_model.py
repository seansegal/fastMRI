import torch
from torch import nn
from torch.nn import functional as F

def get_norm_layer(norm, out_chans):

    if norm == 'instance2d':
        return nn.InstanceNorm2d(out_chans)
    elif norm == 'batch2d':
        return nn.BatchNorm2d(out_chans)
    if norm == 'instance1d':
        return nn.InstanceNorm1d(out_chans)
    elif norm == 'batch1d':
        return nn.BatchNorm1d(out_chans)
    elif norm == 'layer1d':
        return nn.LayerNorm(out_chans) 
    else:
        raise NotImplementedError

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob, non_linearity, norm='instance'):
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
            get_norm_layer(norm, out_chans),
            non_linearity,
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            get_norm_layer(norm, out_chans),
            non_linearity,
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


class GanGenerator(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.norm = 'instance2d'
        # self.non_linearity = nn.LeakyReLU()
        self.non_linearity = nn.ReLU()
        # self.pool = F.avg_pool2d
        self.pool = F.max_pool2d
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob, 
                                                            self.non_linearity, norm=self.norm)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob, self.non_linearity, norm=self.norm)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob, self.non_linearity, norm=self.norm)

        self.up_sample_layers = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob, self.non_linearity, norm=self.norm)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob, self.non_linearity, norm=self.norm)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            get_norm_layer(self.norm, ch // 2),
            self.non_linearity,
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            get_norm_layer(self.norm, out_chans),
            self.non_linearity,
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = self.pool(output, kernel_size=2)

        output = self.conv(output)

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            skip_tensor = stack.pop()
            col_diff = skip_tensor.shape[-1] - output.shape[-1]
            row_diff = skip_tensor.shape[-2] - output.shape[-2]
            assert col_diff <= 1
            assert row_diff <= 1
            padding = (col_diff, 0, row_diff, 0)
            output_padded = F.pad(output, padding)
            output = torch.cat([output_padded, skip_tensor], dim=1)
            output = layer(output)
        return self.conv2(output)


class GanDiscriminator(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, chans, num_pool_layers, drop_prob):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.in_features_fc = chans * 2**(num_pool_layers-1)
        self.norm = 'instance2d'
        self.fc_norm = 'layer1d'
        self.non_linearity = nn.LeakyReLU()
        # self.pool = F.avg_pool2d
        self.pool = F.max_pool2d
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob, self.non_linearity, norm=self.norm)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob, self.non_linearity, norm=self.norm)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob, self.non_linearity, norm=self.norm)
        self.fc = nn.Sequential(nn.Linear(self.in_features_fc, self.in_features_fc // 2),
                                get_norm_layer(self.fc_norm, self.in_features_fc // 2),
                                self.non_linearity,
                                nn.Linear(self.in_features_fc // 2, self.in_features_fc // 4),
                                get_norm_layer(self.fc_norm, self.in_features_fc // 4),
                                self.non_linearity,
                                nn.Linear(self.in_features_fc // 4, 1)
                            )


    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input 

        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = self.pool(output, kernel_size=2)
        # Flatten 
        output = self.pool(output, 10).view(output.shape[0], -1)

        # Fully Connected Layers
        output = self.fc(output).squeeze(1)
        return output
