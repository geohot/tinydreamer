from typing import List
from tinygrad import Tensor, nn
from einops import rearrange

class Downsample:
  def __init__(self, num_channels: int) -> None:
    self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=2, stride=2, padding=0)
  def __call__(self, x: Tensor) -> Tensor: return self.conv(x)

class Upsample:
  def __init__(self, num_channels: int) -> None:
    self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
  def __call__(self, x: Tensor) -> Tensor:
    # TODO: is this fast?
    # AssertionError: only supports linear interpolate
    #x = x.interpolate([s*2 for s in x.size()], mode="nearest")
    x = rearrange(x, 'b c h w -> b c h 1 w 1').expand(x.shape[0], x.shape[1], x.shape[2], 2, x.shape[3], 2) \
      .reshape(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2)
    return self.conv(x)

class ResidualBlock:
  def __init__(self, in_channels: int, out_channels: int, num_groups_norm: int = 32) -> None:
    self.f = [
      nn.GroupNorm(num_groups_norm, in_channels),
      Tensor.silu,
      nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
      nn.GroupNorm(num_groups_norm, out_channels),
      Tensor.silu,
      nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
    ]
    self.skip_projection = (lambda x: x) if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
  def __call__(self, x: Tensor) -> Tensor: return self.skip_projection(x) + x.sequential(self.f)

class FrameDecoder:
  def __init__(self):
    self.decoder = [
      nn.Conv2d(84, 256, kernel_size=3, stride=1, padding=1),
      ResidualBlock(256, 128), Upsample(128),
      ResidualBlock(128, 128), Upsample(128),
      ResidualBlock(128, 64),
      ResidualBlock(64, 64), Upsample(64),
      ResidualBlock(64, 64),
      nn.GroupNorm(num_groups=32, num_channels=64),
      Tensor.silu,
      nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
    ]

  def __call__(self, x:Tensor) -> Tensor:
    b, t, _, _, _ = x.size()
    x = rearrange(x, 'b t c h w -> (b t) c h w')
    x = x.sequential(self.decoder)
    x = rearrange(x, '(b t) c h w -> b t c h w', b=b, t=t)
    return x

class FrameEncoder:
  def __init__(self, channels: List[int]):
    self.encoder = [
      nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
      ResidualBlock(channels[1], channels[1]), Downsample(channels[1]),
      ResidualBlock(channels[1], channels[1]),
      ResidualBlock(channels[1], channels[2]), Downsample(channels[2]),
      ResidualBlock(channels[2], channels[2]), Downsample(channels[2]),
      ResidualBlock(channels[2], channels[3]),
      nn.GroupNorm(num_groups=32, num_channels=channels[3]),
      Tensor.silu,
      nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
    ]

  def __call__(self, x: Tensor) -> Tensor:
    b, t, _, _, _ = x.size()
    x = rearrange(x, 'b t c h w -> (b t) c h w')
    x = x.sequential(self.encoder)
    x = rearrange(x, '(b t) c h w -> b t c h w', b=b, t=t)
    return x