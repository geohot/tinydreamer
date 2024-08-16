#!/usr/bin/env python3
import gymnasium as gym
from model import Model, CnnLstmActorCritic, MaxAndSkipEnv, preprocess, EMBED_DIM
from tinygrad import nn, Tensor, GlobalCounters
from einops import rearrange

BS = 16

import pygame

screen = None
SCALE = 4
def draw(x:Tensor):
  global screen
  if screen is None:
    pygame.init()
    screen = pygame.display.set_mode((x.shape[1]*SCALE, x.shape[2]*SCALE))
  img = x.permute(2,1,0)
  surf = pygame.surfarray.make_surface((img*256).cast('uint8').repeat_interleave(SCALE, 0).repeat_interleave(SCALE, 1).numpy())
  screen.blit(surf, (0, 0))
  pygame.display.flip()

if __name__ == "__main__":
  model = Model()
  nn.state.load_state_dict(model, nn.state.torch_load("last.pt"))
  ac = CnnLstmActorCritic(6)

  env = MaxAndSkipEnv(gym.make('PongNoFrameskip-v4'))
  obs, info = env.reset()
  img_0 = preprocess(obs).expand(BS, -1, -1, -1, -1)

  transformer_tokens = Tensor.zeros(BS, 0, EMBED_DIM)
  ac.reset()
  for j in range(25):
    GlobalCounters.reset()
    print(img_0.shape)
    draw(rearrange(img_0, "(bw bh) 1 c w h -> c (bw w) (bh h)", bw=4))

    ac_out = ac(img_0)
    sampled_actions = ac_out.logits_actions.exp().softmax().squeeze(1).multinomial()

    frame_emb = img_0.sequential(model.world_model.frame_cnn)[:, 0]
    act_emb = model.world_model.act_emb(sampled_actions)
    transformer_tokens = transformer_tokens.cat(frame_emb, act_emb, dim=1).contiguous()
    latents = []
    for i in range(4):
      out = model.world_model.transformer(transformer_tokens)
      logits_latents = model.world_model.head_latents(out[:, -1:])
      latent = logits_latents.exp().softmax().squeeze(1).multinomial()
      latents.append(latent)
      transformer_tokens = transformer_tokens.cat(model.world_model.latents_emb(latent), dim=1)
    latents = model.tokenizer.quantizer.embed_tokens(Tensor.cat(*latents, dim=1)).unsqueeze(1)
    qq = rearrange(latents, 'b t (h w) (k l e) -> b t e (h k) (w l)',
                   h=model.tokenizer.tokens_grid_res, k=model.tokenizer.token_res, l=model.tokenizer.token_res)
    img_0 = model.tokenizer.decode(img_0, sampled_actions, qq, should_clamp=True)
