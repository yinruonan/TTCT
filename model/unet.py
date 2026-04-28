import torch.nn as nn
import torch


class UNet2D(nn.Module):
    def __init__(self, in_channels=1):
        super(UNet2D, self).__init__()
        self.conv1 = self.get_block(in_channels, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = self.get_block(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = self.get_block(32, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = self.get_block(64, 128)

        self.upsampling_1 = nn.Upsample(scale_factor=2)
        self.conv5 = self.get_block(192, 64)

        self.upsampling_2 = nn.Upsample(scale_factor=2)
        self.conv6 = self.get_block(64 + 32, 32)

        self.upsampling_3 = nn.Upsample(scale_factor=2)
        self.conv7 = self.get_block(32 + 16, 16)

        self.conv8 = nn.Conv2d(16, 2, kernel_size=(1, 1))


    def get_block(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):
        conv_layer_1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        bn1 = nn.BatchNorm2d(out_channels)
        act_layer_1 = nn.ReLU()

        conv_layer_2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        bn2 = nn.BatchNorm2d(out_channels)
        act_layer_2 = nn.ReLU()

        block = nn.Sequential(
            conv_layer_1,
            bn1, 
            act_layer_1,
            conv_layer_2,
            bn2,
            act_layer_2,
        )
        return block

    def forward(self, x, **kwargs):
        x1 = self.conv1(x)  # 128
        p1 = self.pool1(x1)  # 64
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)  # 32
        x3 = self.conv3(p2)
        p3 = self.pool3(x3)  # 16

        x4 = self.conv4(p3)  # 16

        up5 = torch.cat([self.upsampling_1(x4), x3], dim=1)
        x5 = self.conv5(up5)

        up6 = torch.cat([self.upsampling_2(x5), x2], dim=1)
        x6 = self.conv6(up6)

        up7 = torch.cat([self.upsampling_2(x6), x1], dim=1)
        x7 = self.conv7(up7)
        out = {
            'logits': self.conv8(x7)
        }
        return out
