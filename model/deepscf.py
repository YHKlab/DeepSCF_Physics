import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import get_encoder
from .decoder import get_decoder
from .initial import get_initial

class DeepSCF(nn.Module):
    def __init__(self, args):
        super(DeepSCF, self).__init__()

        # Data distributuin (standard)
        self.mean = args.rho0_mean
        self.std = args.rho0_std
        self.rho0_index = args.rho0_index

        # Convolutional layer info.
        layers = args.layers
        size = args.kernal_size
        padding = args.padding_mode

        # Input features
        input_layers = args.input_layers
        self.input_layers = input_layers
        self.input_layers_index = args.input_layers_index

        residual = args.residual  # skip-connection
        self.residual = residual

        # CBAM block
        cbam = args.cbam

        # Depthwise-sperable (DS) layers
        if (args.depthwise_layers != None):
            depthwise = args.depthwise_layers # [1,1,1] => full DS U-Net
        else:
            depthwise = [0,0,0] # [0,0,0] => original U-Net



        # Normalization
        self.normalization = args.normalization

        # Activation functions
        if args.activation_function == 'ReLU':
            activation_function = nn.ReLU(inplace = True)
        elif args.activation_function == 'ELU':
            activation_function = nn.ELU(inplace = True)

        # Pooling methods
        if args.pooling_method == 'average':
            pooling_layer = nn.AvgPool3d(kernel_size=2)
        elif args.pooling_method == 'max':
            pooling_layer = nn.MaxPool3d(kernel_size=2)

        self.initial = get_initial(
                                    in_channels=input_layers,
                                    out_channels=layers,
                                    kernel_size=size,
                                    activation_function=activation_function,
                                    padding_mode=padding
                                   )

        self.final = nn.Sequential(
            nn.Conv3d(in_channels=layers,
                      out_channels=1,
                      kernel_size=(1,1,1),
                      stride=(1,1,1),
                      padding="same",
                      padding_mode=padding,
                      bias=False)
        )

        # Encoder modules
        self.encoder1 = get_encoder(depthwise=depthwise[0],
                                    in_channels=layers,
                                    out_channels=2*layers,
                                    kernel_size=size,
                                    activation_function=activation_function,
                                    pooling_layer=pooling_layer,
                                    padding_mode=padding
                                   )
        self.encoder2 = get_encoder(depthwise=depthwise[1],
                                    in_channels=2*layers,
                                    out_channels=4*layers,
                                    kernel_size=size,
                                    activation_function=activation_function,
                                    pooling_layer=pooling_layer,
                                    padding_mode=padding
                                   )
        self.encoder3 = get_encoder(depthwise=depthwise[2],
                                    in_channels=4*layers,
                                    out_channels=8*layers,
                                    kernel_size=size,
                                    activation_function=activation_function,
                                    pooling_layer=pooling_layer,
                                    padding_mode=padding
                                   )

        # Decoder modules
        self.decoder3 = get_decoder(depthwise=depthwise[2],
                                    cbam=cbam[2],
                                    in_channels=8*layers,
                                    out_channels=4*layers,
                                    kernel_size=size,
                                    activation_function=activation_function,
                                    padding_mode=padding
                                   )
        self.decoder2 = get_decoder(depthwise=depthwise[1],
                                    cbam=cbam[1],
                                    in_channels=4*layers,
                                    out_channels=2*layers,
                                    kernel_size=size,
                                    activation_function=activation_function,
                                    padding_mode=padding
                                   )
        self.decoder1 = get_decoder(depthwise=depthwise[0],
                                    cbam=cbam[0],
                                    in_channels=2*layers,
                                    out_channels=1*layers,
                                    kernel_size=size,
                                    activation_function=activation_function,
                                    padding_mode=padding
                                   )


    def forward(self, input):

        # activation function
        relu = nn.ReLU(inplace=True)

        # reference density
        x = input[:,int(self.rho0_index),:,:,:] * self.std + self.mean
        charge0 = torch.sum(x,(3,2,1))

        # residual
        residual = self.residual

        # custom slicing
        input_layers_index = self.input_layers_index
        if (input_layers_index != None):
            input = input[:,input_layers_index,:,:,:]


        x0 = self.initial(input)

        # encoding layers
        x1 = self.encoder1(x0)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # decoding layers
        out = self.decoder3(x3, x2)
        out = self.decoder2(out, x1)
        out = self.decoder1(out, x0)

        # final layers
        out = self.final(out)
        out = out.squeeze(dim=1)

        if (residual):
            out = relu(out+x)
        else:
            out = relu(out)

        # normalization
        if (self.normalization):
            charge = torch.sum(out,(3,2,1))
            scale = (charge0/charge).view(-1,1,1,1)
            return scale * out
        else:
            return out
