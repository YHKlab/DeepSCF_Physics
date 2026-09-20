import torch
import torch.nn as nn
import torch.nn.functional as F
from .convolution import depthConv3d, depthConvTranspose3d, CBAM

def get_decoder(depthwise, cbam, in_channels, out_channels, kernel_size, activation_function, padding_mode):

    if (depthwise):
        return depthDecoder(in_channels, out_channels, kernel_size, activation_function, padding_mode, cbam)
    else:
        return Decoder(in_channels, out_channels, kernel_size, activation_function, padding_mode, cbam)


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation_function, padding_mode, cbam):
        super(Decoder, self).__init__()
        size = kernel_size
        padding = padding_mode
        self.cbam = cbam

        # upsampling

        self.upsampling = nn.ConvTranspose3d(in_channels=in_channels,
                                             out_channels=in_channels,
                                             kernel_size=(2,2,2),
                                             stride=(2,2,2),
                                             padding=0,
                                             padding_mode='zeros',
                                             bias=False)

        # convolution
        self.convolution1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels+out_channels,
                      out_channels=out_channels,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode=padding,
                      bias=False),
            activation_function
        )
        self.convolution2 = nn.Sequential(
            nn.Conv3d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode=padding,
                      bias=False),
            activation_function

        )

        if self.cbam:
            self.cbam_block = CBAM(in_channels=out_channels)

    def forward(self, x, x0):

        out = self.upsampling(x)
        out = torch.cat((out, x0), dim=1) # skip connection
        out = self.convolution1(out)
        out = self.convolution2(out)

        if self.cbam:
            out = self.cbam_block(out)

        return out
class depthDecoder(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation_function, padding_mode, cbam):
        super(depthDecoder, self).__init__()
        size = kernel_size
        padding = padding_mode
        self.cbam = cbam

        # upsampling

        self.upsampling = depthConvTranspose3d(in_channels=in_channels,
                                             out_channels=in_channels,
                                             kernel_size=(2,2,2),
                                             stride=(2,2,2),
                                             padding=0,
                                             padding_mode=padding,
                                             bias=False)

        # convolution
        self.convolution1 = nn.Sequential(
            depthConv3d(in_channels=in_channels+out_channels,
                      out_channels=out_channels,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode=padding,
                      bias=False),
            activation_function
        )
        self.convolution2 = nn.Sequential(
            depthConv3d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(size,size,size),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode=padding,
                      bias=False),
            activation_function

        )

        if self.cbam:
            self.cbam_block = CBAM(in_channels=out_channels)

    def forward(self, x, x0):

        out = self.upsampling(x)
        out = torch.cat((out, x0), dim=1) # skip connection
        out = self.convolution1(out)
        out = self.convolution2(out)

        if self.cbam:
            out = self.cbam_block(out)

        return out
