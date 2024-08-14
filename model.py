from typing import List, Dict, Optional, Tuple
import math
from dataclasses import dataclass
from tinygrad import Tensor, nn, TinyJit
from einops import rearrange
import gymnasium as gym
from PIL import Image
import numpy as np

from models.frame import FrameDecoder, FrameEncoder
from models.transformer import EMBED_DIM, TransformerEncoder, Head
from models.quantizer import Quantizer

# TODO: i like torches tensors that include dtype in the type

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

@dataclass
class WorldModelOutput:
  output_sequence: Tensor
  logits_latents: Tensor
  logits_rewards: Tensor
  logits_ends: Tensor

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
    self.lstm = nn.LSTMCell(1024, self.lstm_dim)

  def __call__(self, x:Tensor) -> ActorCriticOutput:
    x = x.sequential(self.cnn)
    self.hx, self.cx = self.lstm(x, (self.hx, self.cx) if self.hx is not None else None)
    logits_actions = rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')
    logits_values = rearrange(self.critic_linear(self.hx), 'b c -> b 1 c')
    return ActorCriticOutput(logits_actions, logits_values)

class Model:
  def __init__(self):
    self.world_model = WorldModel()
    self.tokenizer = Tokenizer()
    self.actor_critic = {"model": CnnLstmActorCritic(6), "target_model": CnnLstmActorCritic(6)}

# TODO: this should be written in tinygrad. tinygrad needs to support NEAREST
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

  # TODO: is this correct with the LSTM
  #@TinyJit
  def get_action(img_0:Tensor) -> Tensor:
    x = model.actor_critic['target_model'](img_0)
    action = x.logits_actions.exp().softmax().flatten().multinomial()
    return action

  for i in range(1000):
    img_0 = preprocess(obs)
    action = get_action(img_0)
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
