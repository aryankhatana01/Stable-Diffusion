import torch.nn as nn
import math
import numpy as np
import torch


class ResBlock(nn.Module):
  def __init__(self, channels, emb_channels, out_channels):
    super(ResBlock, self).__init__()
    self.in_layers = nn.Sequential(
      nn.GroupNorm(32, channels),
      nn.SiLU(),
      nn.Conv2d(channels, out_channels, 3, padding=1)
    )
    self.emb_layers = nn.Sequential(
      nn.SiLU(),
      nn.Linear(emb_channels, out_channels)
    )
    self.out_layers = nn.Sequential(
      nn.GroupNorm(32, out_channels),
      nn.SiLU(),
      lambda x: x,
      nn.Conv2d(out_channels, out_channels, 3, padding=1)
    )
    self.skip_connection = nn.Conv2d(channels, out_channels, 1) if channels != out_channels else lambda x: x

  def forward(self, x, emb):
    h = self.in_layers(x)
    emb_out = self.emb_layers(emb)
    h = h + emb_out.reshape(*emb_out.shape, 1, 1)
    h = self.out_layers(h)
    ret = self.skip_connection(x) + h
    return ret


class CrossAttention(nn.Module):
  def __init__(self, query_dim, context_dim, n_heads, d_head):
    super(CrossAttention, self).__init__()
    self.to_q = nn.Linear(query_dim, n_heads*d_head, bias=False)
    self.to_k = nn.Linear(context_dim, n_heads*d_head, bias=False)
    self.to_v = nn.Linear(context_dim, n_heads*d_head, bias=False)
    self.scale = d_head ** -0.5
    self.num_heads = n_heads
    self.head_size = d_head
    self.to_out = nn.Sequential(nn.Linear(n_heads*d_head, query_dim))

  def forward(self, x, context=None):
    context = x if context is None else context
    q,k,v = self.to_q(x), self.to_k(context), self.to_v(context)
    q = q.reshape(x.shape[0], -1, self.num_heads, self.head_size).permute(0,2,1,3)  # (bs, num_heads, time, head_size)
    k = k.reshape(x.shape[0], -1, self.num_heads, self.head_size).permute(0,2,3,1)  # (bs, num_heads, head_size, time)
    v = v.reshape(x.shape[0], -1, self.num_heads, self.head_size).permute(0,2,1,3)  # (bs, num_heads, time, head_size)

    score = q.dot(k) * self.scale
    weights = score.softmax()                     # (bs, num_heads, time, time)
    attention = weights.dot(v).permute(0,2,1,3)   # (bs, time, num_heads, head_size)

    h_ = attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size))
    return self.to_out(h_)

class GEGLU(nn.Module):
  def __init__(self, dim_in, dim_out):
    super(GEGLU, self).__init__()
    self.proj = nn.Linear(dim_in, dim_out * 2)
    self.dim_out = dim_out

  def forward(self, x):
    x, gate = self.proj(x).chunk(2, dim=-1)
    return x * nn.GELU()(gate)

class FeedForward(nn.Module):
  def __init__(self, dim, mult=4):
    super(FeedForward, self).__init__()
    self.net = nn.Sequential(
      GEGLU(dim, dim*mult),
      lambda x: x,
      nn.Linear(dim*mult, dim)
    )

  def forward(self, x):
    return self.net(x)

class BasicTransformerBlock(nn.Module):
  def __init__(self, dim, context_dim, n_heads, d_head):
    super(BasicTransformerBlock, self).__init__()
    self.attn1 = CrossAttention(dim, dim, n_heads, d_head)
    self.ff = FeedForward(dim)
    self.attn2 = CrossAttention(dim, context_dim, n_heads, d_head)
    self.norm1 = nn.LayerNorm(dim)
    self.norm2 = nn.LayerNorm(dim)
    self.norm3 = nn.LayerNorm(dim)

  def forward(self, x, context=None):
    x = self.attn1(self.norm1(x)) + x
    x = self.attn2(self.norm2(x), context=context) + x
    x = self.ff(self.norm3(x)) + x
    return x

class SpatialTransformer(nn.Module):
  def __init__(self, channels, context_dim, n_heads, d_head):
    super(SpatialTransformer, self).__init__()
    self.norm = nn.GroupNorm(32, channels)
    assert channels == n_heads * d_head
    self.proj_in = nn.Conv2d(channels, n_heads * d_head, 1)
    self.transformer_blocks = [BasicTransformerBlock(channels, context_dim, n_heads, d_head)]
    self.proj_out = nn.Conv2d(n_heads * d_head, channels, 1)

  def forward(self, x, context=None):
    b, c, h, w = x.shape
    x_in = x
    x = self.norm(x)
    x = self.proj_in(x)
    x = x.reshape(b, c, h*w).permute(0,2,1)
    for block in self.transformer_blocks:
      x = block(x, context=context)
    x = x.permute(0,2,1).reshape(b, c, h, w)
    ret = self.proj_out(x) + x_in
    return ret

class Downsample(nn.Module):
  def __init__(self, channels):
    super(Downsample, self).__init__()
    self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

  def forward(self, x):
    return self.op(x)

class Upsample(nn.Module):
  def __init__(self, channels):
    super(Upsample, self).__init__()
    self.conv = nn.Conv2d(channels, channels, 3, padding=1)

  def forward(self, x):
    bs,c,py,px = x.shape
    x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
    return self.conv(x)

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = np.exp(-math.log(max_period) * np.arange(0, half, dtype=np.float32) / half)
    args = timesteps * freqs
    embedding = np.concatenate([np.cos(args), np.sin(args)])
    return torch.tensor(embedding).reshape(1, -1)