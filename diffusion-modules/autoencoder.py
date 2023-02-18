import torch.nn as nn
from resnet import ResnetBlock
from mid import Mid
import torch.nn.functional as F

def swish(x):
    return x * F.sigmoid(x)

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    sz = [(128, 256), (256, 512), (512, 512), (512, 512)]
    self.conv_in = nn.Conv2d(4,512,3, padding=1)
    self.mid = Mid(512)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[1], s[0]),
         ResnetBlock(s[0], s[0]),
         ResnetBlock(s[0], s[0])]})
      if i != 0: arr[-1]['upsample'] = {"conv": nn.Conv2d(s[0], s[0], 3, padding=1)}
    self.up = arr

    self.norm_out = nn.GroupNorm(32, 128)
    self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

  def forward(self, x):
    x = self.conv_in(x)
    x = self.mid(x)

    for l in self.up[::-1]:
      print("decode", x.shape)
      for b in l['block']: x = b(x)
      if 'upsample' in l:
        bs,c,py,px = x.shape
        x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, 2, px, 2).reshape(bs, c, py*2, px*2)
        x = l['upsample']['conv'](x)
      # x.realize()

    return self.conv_out(swish(self.norm_out(x)))


class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    sz = [(128, 128), (128, 256), (256, 512), (512, 512)]
    self.conv_in = nn.Conv2d(3,128,3, padding=1)

    arr = []
    for i,s in enumerate(sz):
      arr.append({"block":
        [ResnetBlock(s[0], s[1]),
         ResnetBlock(s[1], s[1])]})
      if i != 3: arr[-1]['downsample'] = {"conv": nn.Conv2d(s[1], s[1], 3, stride=2, padding=1)}
    self.down = arr

    self.mid = Mid(512)
    self.norm_out = nn.GroupNorm(32, 512)
    self.conv_out = nn.Conv2d(512, 8, 3, padding=1)

  def forward(self, x):
    x = self.conv_in(x)

    for l in self.down:
      print("encode", x.shape)
      for b in l['block']: x = b(x)
      if 'downsample' in l: x = l['downsample']['conv'](x)

    x = self.mid(x)
    return self.conv_out(swish(self.norm_out(x)))

class AutoencoderKL(nn.Module):
  def __init__(self):
    super(AutoencoderKL, self).__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()
    self.quant_conv = nn.Conv2d(8, 8, 1)
    self.post_quant_conv = nn.Conv2d(4, 4, 1)

  def forward(self, x):
    latent = self.encoder(x)
    latent = self.quant_conv(latent)
    latent = latent[:, 0:4]  # only the means
    print("latent", latent.shape)
    latent = self.post_quant_conv(latent)
    return self.decoder(latent)

if __name__ == "__main__":
  import torch
  x = torch.randn(1,3,256,256)
  model = AutoencoderKL()
  y = model(x)
  print(y.shape)