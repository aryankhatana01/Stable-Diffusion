import torch.nn as nn
import torch.nn.functional as F

class AttnBlock(nn.Module):
  def __init__(self, in_channels):
    super(AttnBlock, self).__init__()
    self.norm = nn.GroupNorm(32, in_channels)
    self.q = nn.Conv2d(in_channels, in_channels, 1)
    self.k = nn.Conv2d(in_channels, in_channels, 1)
    self.v = nn.Conv2d(in_channels, in_channels, 1)
    self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

  def forward(self, x):
    h_ = self.norm(x)
    q,k,v = self.q(h_), self.k(h_), self.v(h_)

    # compute attention
    b,c,h,w = q.shape
    q = q.reshape(b,c,h*w)
    q = q.permute(0,2,1)   # b,hw,c
    k = k.reshape(b,c,h*w) # b,c,hw
    w_ = q @ k
    w_ = w_ * (c**(-0.5))
    w_ = F.softmax(w_)

    v = v.reshape(b,c,h*w)
    w_ = w_.permute(0,2,1)
    h_ = v @ w_
    h_ = h_.reshape(b,c,h,w)

    return x + self.proj_out(h_)
