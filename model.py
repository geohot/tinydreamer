from typing import List, Dict
import math
from dataclasses import dataclass
from tinygrad import Tensor,nn
from einops import rearrange
import gymnasium as gym
from PIL import Image
import numpy as np
# TODO: i like torches tensors that include dtype in the type

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

@dataclass
class QuantizerOutput:
  q: Tensor
  tokens: Tensor
  loss: Dict[str, Tensor]
  metrics: Dict[str, float]

class Quantizer:
  def __init__(self, codebook_size: int, codebook_dim: int, input_dim: int):
    assert math.log2(codebook_size).is_integer()
    self.pre_quant_proj = nn.Linear(input_dim, codebook_dim)
    self.post_quant_proj = nn.Linear(codebook_dim, input_dim)
    self.codebook = Tensor.uniform(codebook_size, codebook_dim, low=-1.0 / codebook_size, high=1.0 / codebook_size)

  def __call__(self, z:Tensor) -> QuantizerOutput:
    z = self.pre_quant_proj(z)
    b, k = z.size(0), z.size(2)
    z = rearrange(z, 'b t k e -> (b t k) e')

    cosine_similarity = Tensor.einsum('n e, c e -> n c', z, self.codebook)
    tokens = cosine_similarity.argmax(axis=-1)  # TODO: support both axis and dim
    q = self.codebook[tokens]

    losses = {'commitment_loss': 0.02 * (z - q.detach()).pow(2).mean()}
    metrics = {}

    q = z + (q - z).detach()
    q = self.post_quant_proj(q)

    q = rearrange(q, '(b t k) e -> b t k e', b=b, k=k)
    tokens = rearrange(tokens, '(b t k) -> b t k', b=b, k=k)

    return QuantizerOutput(q, tokens, losses, metrics)

class Tokenizer:
  def __init__(self):
    self.encoder_act_emb = nn.Embedding(6, 4096)
    self.decoder_act_emb = nn.Embedding(6, 256)
    self.frame_cnn = FrameEncoder([3,32,64,128,16])
    self.encoder = FrameEncoder([7,64,128,256,64])
    self.decoder = FrameDecoder()
    self.quantizer = Quantizer(codebook_size=1024, codebook_dim=64, input_dim=1024)

    # guessed to make dims match
    self.token_res = 4
    self.tokens_grid_res = 2

  # need typing
  def encode(self, x1: Tensor, a: Tensor, x2: Tensor) -> Tensor:
    a_emb = rearrange(self.encoder_act_emb(a), 'b t (h w) -> b t 1 h w', h=x1.size(3))
    encoder_input = Tensor.cat(x1, a_emb, x2, dim=2)
    return self.encoder(encoder_input)

  def decode(self, x1: Tensor, a: Tensor, q2: Tensor, should_clamp: bool = False) -> Tensor:
    x1_emb = self.frame_cnn(x1)
    a_emb = rearrange(self.decoder_act_emb(a), 'b t (c h w) -> b t c h w', c=4, h=x1_emb.size(3))
    print(x1_emb.shape, a_emb.shape, q2.shape)

    decoder_input = Tensor.cat(x1_emb, a_emb, q2, dim=2)
    r = self.decoder(decoder_input)
    r = r.clamp(0, 1).mul(255).round().div(255) if should_clamp else r
    return r

  def encode_decode(self, x1: Tensor, a: Tensor, x2: Tensor) -> Tensor:
    z = self.encode(x1, a, x2)
    z = rearrange(z, 'b t c (h k) (w l) -> b t (h w) (k l c)', k=self.token_res, l=self.token_res)
    q = rearrange(self.quantizer(z).q, 'b t (h w) (k l e) -> b t e (h k) (w l)', h=self.tokens_grid_res, k=self.token_res, l=self.token_res)
    r = self.decode(x1, a, q, should_clamp=True)
    return r

  def __call__(self, x: Tensor) -> Tensor:
    return self.frame_cnn(x)

# TODO: this should be in tinygrad
def preprocess(obs, size=(64, 64)):
  image = Image.fromarray(obs).resize(size, Image.NEAREST)
  return Tensor(np.array(image), dtype='float32').permute(2,0,1).reshape(1,1,3,64,64) / 256.0

if __name__ == "__main__":
  env = gym.make('ALE/Pong-v5')
  obs, info = env.reset()

  model = Tokenizer()

  # scp t18:~/build/delta-iris/outputs/2024-08-13/20-34-53/checkpoints/last.pt .
  dat = nn.state.torch_load("last.pt")
  wm = {}
  for k,v in dat.items():
    if k.startswith("tokenizer.") or True:
      print(k, v.shape)
      wm[k[len("tokenizer."):]] = v
  print(len(wm))
  for k in nn.state.get_state_dict(model): print(k)
  nn.state.load_state_dict(model, wm)

  img_0 = preprocess(obs)

  import matplotlib.pyplot as plt

  for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    img_1 = preprocess(obs)
    out = model.encode_decode(img_0, Tensor([action]), img_1)

    print(out)
    pred = out[0, 0].permute(1,2,0).numpy()
    plt.imshow(Tensor.cat(*[x[0, 0].permute(1,2,0) for x in [img_0, img_1, out]], dim=1).numpy())
    plt.draw()
    plt.pause(0.01)
    img_0 = img_1
