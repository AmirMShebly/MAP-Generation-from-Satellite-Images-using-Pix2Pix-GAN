import torch
import torch.nn as nn

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
                                  nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2),)


    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
  def __init__(self, in_channels=3):
    super().__init__()
    self.initial_block = nn.Sequential(
      nn.Conv2d(in_channels*2, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
      nn.LeakyReLU(0.2),
    )

    self.conv1 = Conv_block(64, 128, stride=2)
    self.conv2 = Conv_block(128, 256, stride=2)
    self.conv3 = Conv_block(256, 512, stride=1)
    self.final_conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

  def forward(self, x, y):
    x = torch.cat([x, y], dim=1)
    x = self.initial_block(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    return self.final_conv(x)

