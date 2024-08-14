from typing import List, Dict, Optional, Tuple
import math
from dataclasses import dataclass
from tinygrad import Tensor, nn, TinyJit
from einops import rearrange
import gymnasium as gym
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from models.frame import FrameDecoder, FrameEncoder
from models.transformer import EMBED_DIM, TransformerEncoder, Head
from models.quantizer import Quantizer, QuantizerOutput

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

  #def __call__(self, x: Tensor) -> Tensor:
  #  return self.frame_cnn(x)

@dataclass
class WorldModelOutput:
  output_sequence: Tensor
  logits_latents: Tensor
  logits_rewards: Tensor
  logits_ends: Tensor

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
  env = gym.make('PongNoFrameskip-v4') #, render_mode="human")
  obs, info = env.reset()

  model = Model()

  # scp t18:~/build/delta-iris/outputs/2024-08-13/20-34-53/checkpoints/last.pt .
  dat = nn.state.torch_load("last.pt")
  for k,v in dat.items(): print(k, v.shape)
  nn.state.load_state_dict(model, dat)

  act = 5
  obs, reward, terminated, truncated, info = env.step(act)
  img_0 = preprocess(obs)
  obs, reward, terminated, truncated, info = env.step(0)
  img_1 = preprocess(obs)

  import pygame
  pygame.init()
  screen = pygame.display.set_mode((64*8, 64*8))

  def draw(x:Tensor):
    surf = pygame.surfarray.make_surface((x[0, 0].permute(2,1,0)*256).cast('uint8').repeat_interleave(8, 0).repeat_interleave(8, 1).numpy())
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
        if event.key == pygame.K_w: return 2
        if event.key == pygame.K_s: return 5

  # roll out down

  imgs = [img_0]
  transformer_tokens = Tensor.zeros(1, 0, EMBED_DIM)
  #for i in range(10):
  while 1:
    draw(img_0)
    act = getkey()
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
    img_1_pred = model.tokenizer.decode(img_0, Tensor([[act]]), qq, should_clamp=True)
    imgs.append(img_1_pred)
    img_0 = img_1_pred
  #plt.imshow(Tensor.cat(*[x[0, 0].permute(1,2,0) for x in imgs], dim=1).numpy())
  #plt.show()

  exit(0)


  real_q = model.tokenizer(img_0, Tensor([[act]]), img_1)
  print("real tokens", real_q.tokens.numpy())

  frames_emb = img_0.sequential(model.world_model.frame_cnn)
  act_tokens_emb = rearrange(model.world_model.act_emb(Tensor([[act]])), 'b t e -> b t 1 e')
  print(frames_emb.shape, act_tokens_emb.shape)
  tokens = Tensor.cat(frames_emb, act_tokens_emb, dim=2)[:, 0]
  latents = []
  for i in range(4):
    out = model.world_model.transformer(tokens)
    logits_latents = model.world_model.head_latents(out[:, -1:], 0, 0)[0]
    latent = logits_latents.exp().softmax().multinomial().flatten()
    latents.append(latent)
    emb_latent = model.world_model.latents_emb(latent)
    tokens = tokens.cat(emb_latent, dim=1)

  #tokens = real_q.tokens.flatten()
  tokens = Tensor.cat(*latents)
  print("tokens", tokens.shape, tokens.numpy())
  latents = model.tokenizer.quantizer.embed_tokens(tokens).reshape((1, 1, 4, 1024))

  qq = rearrange(latents, 'b t (h w) (k l e) -> b t e (h k) (w l)',
                 h=model.tokenizer.tokens_grid_res, k=model.tokenizer.token_res, l=model.tokenizer.token_res)
  dec = model.tokenizer.decode(img_0, Tensor([[act]]), qq, should_clamp=True)

  print(dec)

  plt.imshow(Tensor.cat(*[x[0, 0].permute(1,2,0) for x in [img_0, img_1, dec]], dim=1).numpy())
  plt.show()

  #encoded = model.tokenizer(img_0, Tensor([[act]]), img_1)
  #print(encoded.tokens.shape, encoded.tokens.dtype)


  exit(0)

  act = model.world_model.act_emb(Tensor([0]))
  print(act.shape)

  out = model.world_model(act)
  print(out)

  print(out.logits_latents.exp().softmax().flatten().multinomial().numpy())
  print(out.logits_rewards.exp().softmax().numpy())
  print(out.logits_ends.exp().softmax().numpy())

  exit(0)

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
