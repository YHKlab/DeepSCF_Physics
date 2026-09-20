import torch
import torch.nn as nn
import torch.nn.functional as F

class depthConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, padding_mode, bias):
        super(depthConv3d, self).__init__()

    # Depthwise Seperable Convolution
        self.depthConv3d = nn.Sequential(
            # Define Depthwise Convolution
            nn.Conv3d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      groups=in_channels,  # Split input channels for depthwise processing
                      stride=stride,
                      padding=padding,
                      padding_mode=padding_mode,
                      bias=bias),

            # Define Pointwise Convolution
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,  # 1x1 kernel for pointwise operation
                      stride=(1, 1, 1),
                      padding=padding,
                      padding_mode=padding_mode,
                      bias=bias)
        )

    def forward(self, x):
        return self.depthConv3d(x)

class depthConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, padding_mode, bias):
        super(depthConvTranspose3d, self).__init__()

    # Depthwise Seperable Convolution
        self.depthConvTranspose3d = nn.Sequential(
            # Define Depthwise Convolution
            nn.ConvTranspose3d(in_channels=in_channels,
                      out_channels=in_channels,
                      kernel_size=kernel_size,
                      groups=in_channels,  # Split input channels for depthwise processing
                      stride=stride,
                      padding=padding,
                      padding_mode="zeros",
                      bias=bias),

            # Define Pointwise Convolution
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1,  # 1x1 kernel for pointwise operation
                      stride=(1, 1, 1),
                      padding=padding,
                      padding_mode=padding_mode,
                      bias=bias)
        )

    def forward(self, x):
        return self.depthConvTranspose3d(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(CBAM, self).__init__()

        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention =  SpatialAttention()

    def forward(self, x):

        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_channels, in_channels // 16, 1),
                                nn.ReLU(),
                                nn.Conv3d(in_channels // 16, in_channels, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv3d(in_channels=2,
                              out_channels=1,
                              kernel_size=7,
                              padding="same",
                              padding_mode="circular",
                              bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


def laplacian(target, device):

    target= target.unsqueeze(dim=1)

    kernel = torch.tensor([[[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                             [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                             [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]],
                              dtype=target.dtype,
                              device = device
                              )
    kernel.requires_grad = False
    out = F.conv3d(target,
                   kernel,
                   padding="same")

    return out.squeeze(dim=1)


def gradient_3point(target, device):
    target = target.unsqueeze(dim=1)  # (batch, channels=1, depth, height, width)

    # 3-point central difference kernel
    k = torch.tensor([-1, 0, 1], dtype=target.dtype) / 2.0

    # Define directional kernels (3D conv filters)
    # Each is 5-point diff along one axis, identity along the others

    # X-direction: shape (1, 1, 1, 1, 5)
    Gx = k.view(1, 1, 1, 1, 3).to(device)

    # Y-direction: shape (1, 1, 1, 5, 1)
    Gy = k.view(1, 1, 1, 3, 1).to(device)

    # Z-direction: shape (1, 1, 5, 1, 1)
    Gz = k.view(1, 1, 3, 1, 1).to(device)

    Gx.requires_grad = False
    Gy.requires_grad = False
    Gz.requires_grad = False

    # 3D convolution with appropriate padding for 5-point stencil
    grad_x = F.conv3d(target, Gx, padding="same")
    grad_y = F.conv3d(target, Gy, padding="same")
    grad_z = F.conv3d(target, Gz, padding="same")

    return grad_x, grad_y, grad_z


def gradient_5point(target, device):
    target = target.unsqueeze(dim=1)  # (batch, channels=1, depth, height, width)

    # 5-point central difference kernel
    k = torch.tensor([-1, 8, 0, -8, 1], dtype=target.dtype) / 12.0

    # Define directional kernels (3D conv filters)
    # Each is 5-point diff along one axis, identity along the others

    # X-direction: shape (1, 1, 1, 1, 5)
    Gx = k.view(1, 1, 1, 1, 5).to(device)

    # Y-direction: shape (1, 1, 1, 5, 1)
    Gy = k.view(1, 1, 1, 5, 1).to(device)

    # Z-direction: shape (1, 1, 5, 1, 1)
    Gz = k.view(1, 1, 5, 1, 1).to(device)

    Gx.requires_grad = False
    Gy.requires_grad = False
    Gz.requires_grad = False

    # 3D convolution with appropriate padding for 5-point stencil
    grad_x = F.conv3d(target, Gx, padding="same")
    grad_y = F.conv3d(target, Gy, padding="same")
    grad_z = F.conv3d(target, Gz, padding="same")

    return grad_x, grad_y, grad_z


def laplacian_3point(target, device):
    target = target.unsqueeze(dim=1)  # (batch, channels=1, depth, height, width)

    # 3-point second derivative kernel
    k = torch.tensor([1, -2, 1], dtype=target.dtype)

    # X-direction: shape (1, 1, 1, 1, 5)
    Lx = k.view(1, 1, 1, 1, 3).to(device)

    # Y-direction: shape (1, 1, 1, 5, 1)
    Ly = k.view(1, 1, 1, 3, 1).to(device)

    # Z-direction: shape (1, 1, 5, 1, 1)
    Lz = k.view(1, 1, 3, 1, 1).to(device)

    Lx.requires_grad = False
    Ly.requires_grad = False
    Lz.requires_grad = False

    # 3D convolution with appropriate padding for 5-point stencil
    lap_x = F.conv3d(target, Lx, padding="same")
    lap_y = F.conv3d(target, Ly, padding="same")
    lap_z = F.conv3d(target, Lz, padding="same")

    return lap_x + lap_y + lap_z


def laplacian_5point(target, device):
    target = target.unsqueeze(dim=1)  # (batch, channels=1, depth, height, width)

    # 3-point second derivative kernel
    k = torch.tensor([-1, 16, -30, 16, -1], dtype=target.dtype) / 12.0

    # X-direction: shape (1, 1, 1, 1, 5)
    Lx = k.view(1, 1, 1, 1, 5).to(device)

    # Y-direction: shape (1, 1, 1, 5, 1)
    Ly = k.view(1, 1, 1, 5, 1).to(device)

    # Z-direction: shape (1, 1, 5, 1, 1)
    Lz = k.view(1, 1, 5, 1, 1).to(device)

    Lx.requires_grad = False
    Ly.requires_grad = False
    Lz.requires_grad = False

    # 3D convolution with appropriate padding for 5-point stencil
    lap_x = F.conv3d(target, Lx, padding="same")
    lap_y = F.conv3d(target, Ly, padding="same")
    lap_z = F.conv3d(target, Lz, padding="same")

    return lap_x + lap_y + lap_z


class AddCoords3D(nn.Module):
    def __init__(self, with_r: bool = False):
        super(AddCoords3D, self).__init__()
        self.with_r = with_r

    @torch.no_grad()
    def forward(self, x):
        # x: [N, C, D, H, W]
        n, c, d, h, w = x.shape
        device, dtype = x.device, x.dtype

        zz = torch.linspace(-1, 1, steps=d, device=device, dtype=dtype).view(1,1,d,1,1).expand(n,1,d,h,w)
        yy = torch.linspace(-1, 1, steps=h, device=device, dtype=dtype).view(1,1,1,h,1).expand(n,1,d,h,w)
        xx = torch.linspace(-1, 1, steps=w, device=device, dtype=dtype).view(1,1,1,1,w).expand(n,1,d,h,w)

        if self.with_r:
            rr = torch.sqrt(xx*xx + yy*yy + zz*zz)
            return torch.cat([x, xx, yy, zz, rr], dim=1)
        else:
            return torch.cat([x, xx, yy, zz], dim=1)

class CoordConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=0, padding_mode='zeros', bias=True, with_r=False):
        super(CoordConv3d, self).__init__()
        coor_channels = 3 + (1 if with_r else 0)  # x,y,z,(r)
        self.addcoords = AddCoords3D(with_r=with_r)
        self.conv = nn.Conv3d(in_channels + coor_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, padding_mode=padding_mode,
                              bias=bias)

    def forward(self, x):
        x = self.addcoords(x)
        return self.conv(x)
