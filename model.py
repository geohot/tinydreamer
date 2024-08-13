#!/usr/bin/env python3
import gymnasium as gym
import pickle, math
import numpy as np
from tinygrad.helpers import prod
from tinygrad import Tensor, nn
from PIL import Image

class NormConv2d:
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, transp=False):
    self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    self.stride, self.padding, self.transp = stride, padding, transp
    self.scale = Tensor.ones(out_channels)
    self.eps = 1e-6
    scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
    if transp: self.weight = Tensor.uniform(in_channels, out_channels, *self.kernel_size, low=-scale, high=scale)
    else: self.weight = Tensor.uniform(out_channels, in_channels, *self.kernel_size, low=-scale, high=scale)
    self.bias = Tensor.uniform(out_channels, low=-scale, high=scale)

  def __call__(self, x:Tensor) -> Tensor:
    if self.transp: x = x.conv_transpose2d(self.weight, self.bias, padding=self.padding, stride=self.stride, output_padding=1)
    else: x = x.conv2d(self.weight, self.bias, padding=self.padding, stride=self.stride)
    # RMSNorm should work on given channel, not just -1
    x = x * (x.square().mean(1, keepdim=True) + self.eps).rsqrt()
    return x * self.scale.reshape(1, -1, 1, 1)

class Encoder:
  def __init__(self):
    # NOTE: i want padding to support "same"
    self.conv = [
      NormConv2d(1, 64, 5, padding=(2,2)),
      NormConv2d(64, 128, 5, 2, padding=(2,2)),
      NormConv2d(128, 192, 5, 2, padding=(2,2)),
      NormConv2d(192, 256, 5, 2, padding=(2,2)),
      NormConv2d(256, 256, 5, 2, padding=(2,2))]

  def __call__(self, x:Tensor) -> Tensor:
    for c in self.conv: x = c(x).silu()
    return x

class Decoder:
  def __init__(self):
    self.conv = [
      NormConv2d(128, 64, 5, 2, padding=(2,2), transp=True),
      NormConv2d(192, 128, 5, 2, padding=(2,2), transp=True),
      NormConv2d(256, 192, 5, 2, padding=(2,2), transp=True),
      NormConv2d(256, 256, 5, 2, padding=(2,2), transp=True)]
    self.imgout = nn.ConvTranspose2d(64, 1, 5, padding=(2,2))

  def __call__(self, x:Tensor) -> Tensor:
    for c in self.conv[::-1]: x = c(x).silu()
    return self.imgout(x)

def preprocess(obs, size=(64, 64)):
  image = Image.fromarray(obs).resize(size, Image.NEAREST)
  weights = [0.299, 0.587, 1 - (0.299 + 0.587)]
  image = np.tensordot(image, weights, (-1, 0))
  return Tensor(image, dtype='float32').unsqueeze(0)

if __name__ == "__main__":
  env = gym.make('ALE/Pong-v5')
  obs, info = env.reset()

  encoder = Encoder()
  decoder = Decoder()

  dat = pickle.load(open("checkpoint.ckpt", "rb"))
  for k, v in dat['agent'].items():
    print(k, v.shape if hasattr(v, 'shape') else None)
    for s,m,e in [('agent/enc/conv', encoder, True), ('agent/dec/conv', decoder, False)]:
      if k.startswith(s):
        if k.endswith('kernel'): m.conv[int(k.split(s)[1].split("/")[0])].weight.assign(v.transpose(3,2,0,1) if e else v.transpose(2,3,0,1))
        if k.endswith('bias'): m.conv[int(k.split(s)[1].split("/")[0])].bias.assign(v)
        if k.endswith('scale'): m.conv[int(k.split(s)[1].split("/")[0])].scale.assign(v)
    if k == 'agent/dec/imgout/kernel': decoder.imgout.weight.assign(v.transpose(2,3,0,1))
    if k == 'agent/dec/imgout/bias': decoder.imgout.bias.assign(v)

  out = encoder(preprocess(obs, size=(96,96)).unsqueeze(0))
  print(out.shape)
  ret = decoder(out)
  print(ret.shape)
  exit(0)

  import matplotlib.pyplot as plt
  plt.imshow(ret[0, 0].numpy())
  plt.show()
