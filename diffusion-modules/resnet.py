import torch.nn as nn
import torch.nn.functional as F

def swish(x):
  return x * F.sigmoid(x)

class ResnetBlock(nn.Module):
  def __init__(self, in_channels, out_channels=None):
    super(ResnetBlock, self).__init__()
    self.norm1 = nn.GroupNorm(32, in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    self.norm2 = nn.GroupNorm(32, out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

  def forward(self, x):
    h = self.conv1(swish(self.norm1(x)))
    h = self.conv2(swish(self.norm2(h)))
    return self.nin_shortcut(x) + h
