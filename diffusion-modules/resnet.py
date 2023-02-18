import torch.nn as nn

class ResnetBlock(nn.Module):
  def __init__(self, in_channels, out_channels=None):
    super(ResnetBlock, self).__init__()
    self.norm1 = nn.GroupNorm(32, in_channels)
    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    self.norm2 = nn.GroupNorm(32, out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else lambda x: x

  def forward(self, x):
    h = self.conv1(self.norm1(x).swish())
    h = self.conv2(self.norm2(h).swish())
    return self.nin_shortcut(x) + h
