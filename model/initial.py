import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolution import CoordConv3d

def get_initial(in_channels, out_channels, kernel_size, activation_function, padding_mode):

    return Initial(in_channels, out_channels, kernel_size, activation_function, padding_mode)

class Initial(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation_function, padding_mode):
        super(Initial, self).__init__()

        size = kernel_size
        padding = padding_mode

        self.initial = nn.Sequential(
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
        return self.initial(x)

class CoorInitial(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation_function, padding_mode):
        super(CoorInitial, self).__init__()

        size = kernel_size
        padding = padding_mode

        self.initial = nn.Sequential(
            CoordConv3d(in_channels=in_channels,
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
        return self.initial(x)
