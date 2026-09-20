import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolution import depthConv3d, depthConvTranspose3d

def get_encoder(depthwise, in_channels, out_channels, kernel_size, activation_function, pooling_layer, padding_mode):

    if (depthwise):
        return depthEncoder(in_channels, out_channels, kernel_size, activation_function, pooling_layer, padding_mode)
    else:
        return Encoder(in_channels, out_channels, kernel_size, activation_function, pooling_layer, padding_mode)


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation_function, pooling_layer, padding_mode):
        super(Encoder, self).__init__()

        size = kernel_size
        padding = padding_mode

        self.downsampling = nn.Sequential(
            pooling_layer,
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode=padding,
                      bias=False),
            activation_function,


            nn.Conv3d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode=padding,
                      bias=False),
            activation_function
        )

    def forward(self, x):
        return self.downsampling(x)

class depthEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation_function, pooling_layer, padding_mode):
        super(depthEncoder, self).__init__()

        size = kernel_size
        padding = padding_mode

        self.downsampling = nn.Sequential(
            pooling_layer,
            depthConv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode=padding,
                      bias=False),
            activation_function,


            depthConv3d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode=padding,
                      bias=False),
            activation_function
        )

    def forward(self, x):
        return self.downsampling(x)
