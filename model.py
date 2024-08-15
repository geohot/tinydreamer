#!/usr/bin/env python3
from typing import List, Dict, Optional, Tuple
import argparse
import math
from dataclasses import dataclass
from tinygrad import Tensor, nn, TinyJit, dtypes
from einops import rearrange
import gymnasium as gym
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# copied from delta-iris
class MaxAndSkipEnv(gym.Wrapper):
  def __init__(self, env, skip=4):
    """Return only every `skip`-th frame"""
    gym.Wrapper.__init__(self, env)
    assert skip > 0
    # most recent raw observations (for max pooling across time steps)
    self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
    self._skip = skip
    self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

  def step(self, action):
    """Repeat action, sum reward, and max over last observations."""
    total_reward = 0.0
    for i in range(self._skip):
      obs, reward, terminated, truncated, info = self.env.step(action)
      if i == self._skip - 2:
        self._obs_buffer[0] = obs
      if i == self._skip - 1:
        self._obs_buffer[1] = obs
      total_reward += reward
      if terminated or truncated: break
    # Note that the observation on the done=True frame
    # doesn't matter
    self.max_frame = self._obs_buffer.max(axis=0)

    return self.max_frame, total_reward, terminated, truncated, info

  def reset(self, **kwargs): return self.env.reset(**kwargs)


from di.frame import FrameDecoder, FrameEncoder
from di.transformer import EMBED_DIM, TransformerEncoder, Head
from di.quantizer import Quantizer, QuantizerOutput

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

  def __call__(self, x1: Tensor, a: Tensor, x2: Tensor) -> QuantizerOutput:
    z = self.encode(x1, a, x2)
    z = rearrange(z, 'b t c (h k) (w l) -> b t (h w) (k l c)', h=self.tokens_grid_res, w=self.tokens_grid_res)
    return self.quantizer(z)

  # need typing
  def encode(self, x1: Tensor, a: Tensor, x2: Tensor) -> Tensor:
    a_emb = rearrange(self.encoder_act_emb(a), 'b t (h w) -> b t 1 h w', h=x1.size(3))
    encoder_input = Tensor.cat(x1, a_emb, x2, dim=2)
    return self.encoder(encoder_input)

  def decode(self, x1: Tensor, a: Tensor, q2: Tensor, should_clamp: bool = False) -> Tensor:
    x1_emb = self.frame_cnn(x1)
    a_emb = rearrange(self.decoder_act_emb(a), 'b t (c h w) -> b t c h w', c=4, h=x1_emb.size(3))
    decoder_input = Tensor.cat(x1_emb, a_emb, q2, dim=2)
    r = self.decoder(decoder_input)
    r = r.clamp(0, 1).mul(255).round().div(255) if should_clamp else r
    return r

  @TinyJit
  def encode_decode(self, x1: Tensor, a: Tensor, x2: Tensor) -> Tensor:
    z = self.encode(x1, a, x2)
    z = rearrange(z, 'b t c (h k) (w l) -> b t (h w) (k l c)', k=self.token_res, l=self.token_res)
    q = rearrange(self.quantizer(z).q, 'b t (h w) (k l e) -> b t e (h k) (w l)', h=self.tokens_grid_res, k=self.token_res, l=self.token_res)
    r = self.decode(x1, a, q, should_clamp=True)
    return r

  #def __call__(self, x: Tensor) -> Tensor:
  #  return self.frame_cnn(x)

@dataclass
class WorldModelOutput:
  output_sequence: Tensor #[f"b t {EMBED_DIM}", dtypes.float]
  logits_latents: Tensor
  logits_rewards: Tensor #["b t 3", "float"]
  logits_ends: Tensor #["b t 2", "float"]

class WorldModel:
  def __init__(self):
    self.frame_cnn = [FrameEncoder([3,32,64,128,4]), lambda x: rearrange(x, 'b t c h w -> b t 1 (h w c)'), nn.LayerNorm(EMBED_DIM)]
    self.act_emb = nn.Embedding(6, EMBED_DIM)  # embed the action
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
  parser = argparse.ArgumentParser()
  parser.add_argument('--action', choices=['model', 'user', 'random'], default='model',
                      help='Choose the action to perform (default: model)')
  parser.add_argument('--render', choices=['none', 'worldmodel', 'tokenizer'], default='tokenizer',
                      help='Choose the rendering option (default: tokenizer)')
  args = parser.parse_args()

  env = MaxAndSkipEnv(gym.make('PongNoFrameskip-v4')) #, render_mode="human"))
  obs, info = env.reset()

  model = Model()

  # scp t18:~/build/delta-iris/outputs/2024-08-13/20-34-53/checkpoints/last.pt .
  dat = nn.state.torch_load("last.pt")
  nn.state.load_state_dict(model, dat)

  model_state = nn.state.get_state_dict(model)
  for k,v in dat.items():
    if k not in model_state: print("DIDN'T LOAD", k, v.shape)

  import pygame
  pygame.init()
  screen = pygame.display.set_mode((64*8*2, 64*8))

  def draw(x:Tensor):
    img = x[0, 0].permute(2,1,0)
    surf = pygame.surfarray.make_surface((img*256).cast('uint8').repeat_interleave(8, 0).repeat_interleave(8, 1).numpy())
    screen.blit(surf, (0, 0))
    pygame.display.flip()

  def getkey():
    pygame.event.clear()
    while True:
      event = pygame.event.wait()
      if event.type == pygame.QUIT:
        pygame.quit()
      elif event.type == pygame.KEYDOWN:
        print(event.key)
        if event.key == pygame.K_q: return 0
        if event.key == pygame.K_w: return 2
        if event.key == pygame.K_s: return 5

  # TODO: is this correct with the LSTM and TinyJIT
  #@TinyJit
  def get_action(img_0:Tensor) -> Tensor:
    x = model.actor_critic['model'](img_0)
    action = x.logits_actions.exp().softmax().flatten().multinomial()
    return action.item()

  # roll out down
  transformer_tokens = None
  img_0 = None
  while 1:
    cur_img = preprocess(obs)
    if img_0 is None: img_0 = cur_img
    if args.action == "model":
      act = get_action(cur_img)
    elif args.action == "user":
      act = getkey()
    elif args.action == "random":
      act = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(act)
    if terminated or truncated: break

    draw(Tensor.cat(img_0, cur_img, dim=4))

    if args.render == "worldmodel":
      # resync every 20 frames
      if transformer_tokens is None or transformer_tokens.shape[1] >= 20*6:
        img_0 = cur_img
        transformer_tokens = Tensor.zeros(1, 0, EMBED_DIM)

      frames_emb = img_0.sequential(model.world_model.frame_cnn)[:, 0]
      act_tokens_emb = model.world_model.act_emb(Tensor([[act]]))
      print(transformer_tokens.shape, frames_emb.shape, act_tokens_emb.shape)
      transformer_tokens = transformer_tokens.cat(frames_emb, act_tokens_emb, dim=1).contiguous()
      latents = []
      for i in range(4):
        out = model.world_model.transformer(transformer_tokens)
        logits_latents = model.world_model.head_latents(out[:, -1:], 0, 0)[0]
        latent = logits_latents.exp().softmax().multinomial().flatten()
        latents.append(latent)
        transformer_tokens = transformer_tokens.cat(model.world_model.latents_emb(latent), dim=1)
      latents = model.tokenizer.quantizer.embed_tokens(Tensor.cat(*latents)).reshape((1, 1, 4, 1024))
      qq = rearrange(latents, 'b t (h w) (k l e) -> b t e (h k) (w l)',
                    h=model.tokenizer.tokens_grid_res, k=model.tokenizer.token_res, l=model.tokenizer.token_res)
      img_0 = model.tokenizer.decode(img_0, Tensor([[act]]), qq, should_clamp=True)
    elif args.render == "tokenizer":
      img_0 = model.tokenizer.encode_decode(cur_img, Tensor([[act]]), preprocess(obs))
    elif args.render == "none":
      img_0 = preprocess(obs)
