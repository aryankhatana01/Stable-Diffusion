from attention import AttnBlock
from resnet import ResnetBlock
import torch.nn as nn

class Mid(nn.Module):
  def __init__(self, block_in):
    super(Mid, self).__init__()
    self.block_1 = ResnetBlock(block_in, block_in)
    self.attn_1 = AttnBlock(block_in)
    self.block_2 = ResnetBlock(block_in, block_in)

  def forward(self, x):
    return x.sequential([self.block_1, self.attn_1, self.block_2])
