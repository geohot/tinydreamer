from typing import List, Dict, Optional, Tuple
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

EMBED_DIM = 256

class MLPLayer:
  def __init__(self):
    self.ln = nn.LayerNorm(EMBED_DIM)
    self.mlp = [nn.Linear(EMBED_DIM, 4*EMBED_DIM), Tensor.gelu, nn.Linear(4*EMBED_DIM, EMBED_DIM)]
  def __call__(self, x:Tensor): return x + self.ln(x).sequential(self.mlp)

class Attention:
  def __init__(self):
    self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)
    self.num_heads = 4
  def __call__(self, q:Tensor, k:Tensor, v:Tensor) -> Tensor:
    q: Tensor = rearrange(q, 'b q (h e) -> b h q e', h=self.num_heads)
    k = rearrange(k, 'b k (h e) -> b h k e', h=self.num_heads)
    v = rearrange(v, 'b k (h d) -> b h k d', h=self.num_heads)
    y = q.scaled_dot_product_attention(k, v, is_causal=True)
    y = rearrange(y, 'b h q d -> b q (h d)')
    return self.proj(y)

class SelfAttentionLayer:
  def __init__(self):
    self.ln = nn.LayerNorm(EMBED_DIM)
    self.query = nn.Linear(EMBED_DIM, EMBED_DIM)
    self.key = nn.Linear(EMBED_DIM, EMBED_DIM)
    self.value = nn.Linear(EMBED_DIM, EMBED_DIM)
    self.attention = Attention()
  def __call__(self, inputs:Tensor) -> Tensor:
    x = self.ln(inputs)
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    return inputs + self.attention(q, k, v)

class EncoderLayer:
  def __init__(self):
    self.sa = SelfAttentionLayer()
    self.mlp = MLPLayer()
  def __call__(self, x:Tensor) -> Tensor: return self.mlp(self.sa(x))

class TransformerEncoder:
  def __init__(self):
    self.pos_emb = nn.Embedding(156, EMBED_DIM)
    self.ln = nn.LayerNorm(EMBED_DIM)
    self.blocks = [EncoderLayer() for _ in range(3)]
  def __call__(self, x:Tensor) -> Tensor:
    assert x.ndim == 3 and x.size(2) == EMBED_DIM # (B, TK, E)
    y = x + self.pos_emb(Tensor.arange(x.size(1)))
    for i, block in enumerate(self.blocks):
      y = block(y)
    y = self.ln(y)
    return y

@dataclass
class WorldModelOutput:
  output_sequence: Tensor
  logits_latents: Tensor
  logits_rewards: Tensor
  logits_ends: Tensor

class Head:
  def __init__(self, output_dim):
    self.head_module = [
      nn.Linear(EMBED_DIM, EMBED_DIM), Tensor.relu,
      nn.Linear(EMBED_DIM, output_dim)
    ]
  def __call__(self, outputs:Tensor, num_steps, prev_steps):
    return outputs.sequential(self.head_module)

class WorldModel:
  def __init__(self):
    self.frame_cnn = [FrameEncoder([3,32,64,128,4]), lambda x: rearrange(x, 'b t c h w -> b t 1 (h w c)'), nn.LayerNorm(EMBED_DIM)]
    self.act_emb = nn.Embedding(6, EMBED_DIM)
    self.latents_emb = nn.Embedding(1024, EMBED_DIM)
    self.transformer = TransformerEncoder()
    self.head_latents = Head(1024)
    self.head_rewards = Head(3)
    self.head_ends = Head(2)

  def __call__(self, sequence:Tensor) -> WorldModelOutput:
    num_steps, prev_steps = 0, 0
    outputs = self.transformer(sequence)

    logits_latents = self.head_latents(outputs, num_steps, prev_steps)
    logits_rewards = self.head_rewards(outputs, num_steps, prev_steps)
    logits_ends = self.head_ends(outputs, num_steps, prev_steps)

    return WorldModelOutput(outputs, logits_latents, logits_rewards, logits_ends)

# we don't have this in our nn library. TODO: add it?
class LSTMCell:
  def __init__(self):
    self.weight_ih = Tensor.zeros(2048, 1024)
    self.weight_hh = Tensor.zeros(2048, 512)
    self.bias_ih = Tensor.zeros(2048)
    self.bias_hh = Tensor.zeros(2048)
  def __call__(self, x:Tensor, h:Tensor, c:Tensor) -> Tensor:
    gates = x @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
    i, f, g, o = gates.split([512, 512, 512, 512], dim=1)  # TODO: axis
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    new_c = f * c + i * g
    new_h = o * new_c.tanh()
    # contiguous prevents buildup
    return (new_h.contiguous(), new_c.contiguous())

@dataclass
class ActorCriticOutput:
  logits_actions: Tensor
  logits_values: Tensor

class CnnLstmActorCritic:
  def __init__(self, num_actions):
    self.lstm_dim = 512
    self.hx, self.cx = None, None
    self.cnn = [FrameEncoder([3,32,64,128,16]), lambda x: rearrange(x, 'b t c h w -> (b t) (h w c)')]
    self.actor_linear = nn.Linear(self.lstm_dim, num_actions)
    self.critic_linear = nn.Linear(self.lstm_dim, 1)
    self.lstm = LSTMCell()

  def __call__(self, x:Tensor) -> ActorCriticOutput:
    if self.hx is None: self.hx, self.cx = Tensor.zeros(512), Tensor.zeros(512)
    x = x.sequential(self.cnn)
    self.hx, self.cx = self.lstm(x, self.hx, self.cx)
    logits_actions = rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')
    logits_values = rearrange(self.critic_linear(self.hx), 'b c -> b 1 c')
    return ActorCriticOutput(logits_actions, logits_values)

class Model:
  def __init__(self):
    self.world_model = WorldModel()
    self.tokenizer = Tokenizer()
    self.actor_critic = {"model": CnnLstmActorCritic(6), "target_model": CnnLstmActorCritic(6)}

# TODO: this should be written in tinygrad
def preprocess(obs, size=(64, 64)):
  image = Image.fromarray(obs).resize(size, Image.NEAREST)
  return Tensor(np.array(image), dtype='float32').permute(2,0,1).reshape(1,1,3,64,64) / 256.0

if __name__ == "__main__":
  env = gym.make('PongNoFrameskip-v4', render_mode="human")
  obs, info = env.reset()

  model = Model()

  # scp t18:~/build/delta-iris/outputs/2024-08-13/20-34-53/checkpoints/last.pt .
  dat = nn.state.torch_load("last.pt")
  for k,v in dat.items(): print(k, v.shape)
  nn.state.load_state_dict(model, dat)


  for i in range(1000):
    img_0 = preprocess(obs)
    x = model.actor_critic['target_model'](img_0)
    action = x.logits_actions.exp().softmax().flatten().multinomial()
    obs, reward, terminated, truncated, info = env.step(action.item())

  """
  x = rearrange(img_0.sequential(model.world_model.frame_cnn), 'b 1 k e -> b k e')
  a_choice = env.action_space.sample()
  a = model.world_model.act_emb(Tensor([a_choice]))
  # TODO: Tensor.cat((x, a), dim=1) should work also?
  # or should we crack down on all these things...
  # actually probably not on that one, first arg should be Tensor
  output = model.world_model(x.cat(a, dim=1))
  """

  exit(0)


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
