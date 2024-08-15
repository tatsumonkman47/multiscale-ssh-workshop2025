import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    UNet architecture for image segmentation or regression tasks.
    This implementation follows the original UNet structure with some modifications.
    """

    def __init__(self):
        """Initialize the UNet model with encoder, bottleneck, and decoder layers."""
        super(UNet, self).__init__()

        # Encoder (Downsampling path)
        # Each encoder block consists of two convolutional layers followed by max pooling
        self.enc_conv1 = self.conv_block(in_channels=1, out_channels=16)
        self.enc_conv1_1 = self.conv_block(in_channels=16, out_channels=16)
        self.enc_conv2 = self.conv_block(in_channels=16, out_channels=32)
        self.enc_conv2_2 = self.conv_block(in_channels=32, out_channels=32)
        self.enc_conv3 = self.conv_block(in_channels=32, out_channels=64)
        self.enc_conv3_3 = self.conv_block(in_channels=64, out_channels=64)
        self.enc_conv4 = self.conv_block(in_channels=64, out_channels=128)
        self.enc_conv4_4 = self.conv_block(in_channels=128, out_channels=128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (Bridge between encoder and decoder)
        self.bottleneck_conv0 = self.conv_block(in_channels=128, out_channels=256)
        self.bottleneck_conv_0 = self.conv_block(in_channels=256, out_channels=256)

        # Decoder (Upsampling path)
        # Each decoder block consists of an upconvolution followed by two regular convolutions
        self.upconv4 = self.conv_transpose_block(in_channels=256, out_channels=256)
        self.dec_conv4 = self.conv_block(in_channels=256, out_channels=128)
        self.dec_conv4_4 = self.conv_block(in_channels=256, out_channels=128)
        self.upconv3 = self.conv_transpose_block(in_channels=128, out_channels=128)
        self.dec_conv3 = self.conv_block(in_channels=128, out_channels=64)
        self.dec_conv3_3 = self.conv_block(in_channels=128, out_channels=64)
        self.upconv2 = self.conv_transpose_block(in_channels=64, out_channels=64)
        self.dec_conv2 = self.conv_block(in_channels=64, out_channels=32)
        self.dec_conv2_2 = self.conv_block(in_channels=64, out_channels=32)
        self.upconv1 = self.conv_transpose_block(in_channels=32, out_channels=32)
        self.dec_conv1 = self.conv_block(in_channels=32, out_channels=16)
        self.dec_conv1_1 = self.conv_block(in_channels=32, out_channels=16)

        # Final convolution to produce the output
        self.final_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        Create a convolutional block with batch normalization and ReLU activation.
        
        Args:
        - in_channels (int): Number of input channels
        - out_channels (int): Number of output channels
        
        Returns:
        - nn.Sequential: A block of layers (Conv2d, BatchNorm2d, ReLU)
        """
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def conv_transpose_block(self, in_channels, out_channels):
        """
        Create a transpose convolutional block for upsampling.
        
        Args:
        - in_channels (int): Number of input channels
        - out_channels (int): Number of output channels
        
        Returns:
        - nn.Sequential: A block of layers (ConvTranspose2d, BatchNorm2d, ReLU)
        """
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        """
        Forward pass of the UNet.
        
        Args:
        - x (torch.Tensor): Input tensor
        
        Returns:
        - torch.Tensor: Output tensor
        """
        # Encoder path
        x1 = self.enc_conv1_1(self.enc_conv1(x))
        x2 = self.pool(x1)
        x3 = self.enc_conv2_2(self.enc_conv2(x2))
        x4 = self.pool(x3)
        x5 = self.enc_conv3_3(self.enc_conv3(x4))
        x6 = self.pool(x5)
        x7 = self.enc_conv4_4(self.enc_conv4(x6))
        x8 = self.pool(x7)

        # Bottleneck
        p = self.bottleneck_conv_0(self.bottleneck_conv0(x8))

        # Decoder path with skip connections
        u4 = self.upconv4(p)
        u4 = self.dec_conv4(u4)
        # Ensure the upsampled feature map size matches the corresponding encoder feature map
        u4 = F.interpolate(u4, size=x7.size()[2:], mode='bilinear', align_corners=True)
        c4 = torch.cat([u4, x7], dim=1)  # Skip connection
        c4 = self.dec_conv4_4(c4)

        u3 = self.upconv3(c4)
        u3 = self.dec_conv3(u3)
        u3 = F.interpolate(u3, size=x5.size()[2:], mode='bilinear', align_corners=True)
        c3 = torch.cat([u3, x5], dim=1)  # Skip connection
        c3 = self.dec_conv3_3(c3)

        u2 = self.upconv2(c3)
        u2 = self.dec_conv2(u2)
        u2 = F.interpolate(u2, size=x3.size()[2:], mode='bilinear', align_corners=True)
        c2 = torch.cat([u2, x3], dim=1)  # Skip connection
        c2 = self.dec_conv2_2(c2)

        u1 = self.upconv1(c2)
        u1 = self.dec_conv1(u1)
        u1 = F.interpolate(u1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        c1 = torch.cat([u1, x1], dim=1)  # Skip connection
        c1 = self.dec_conv1_1(c1)

        # Final convolution to produce the output
        out = self.final_conv(c1)

        return out
