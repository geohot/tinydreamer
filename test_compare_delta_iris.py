from model import Model, preprocess, MaxAndSkipEnv
from tinygrad import Tensor, nn
import gymnasium as gym
import numpy as np

import sys
sys.path.append("delta-iris/src")

import hydra
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from agent import Agent

import torch
torch.set_grad_enabled(False)

from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from models.actor_critic import ActorCritic

OmegaConf.register_new_resolver("eval", eval)

@hydra.main(config_path="delta-iris/config", config_name="params/atari")
def main(cfg: DictConfig) -> None:
  cfg.params.tokenizer.num_actions = cfg.params.world_model.num_actions = cfg.params.actor_critic.model.num_actions = 6
  agent = Agent(Tokenizer(instantiate(cfg.params.tokenizer)), WorldModel(instantiate(cfg.params.world_model)), ActorCritic(instantiate(cfg.params.actor_critic)))
  agent.load(model_path := Path(__file__).parent / 'last.pt', "cpu", strict=False)

  model = Model()
  dat = nn.state.torch_load(model_path)
  nn.state.load_state_dict(model, dat)

  env = MaxAndSkipEnv(gym.make('PongNoFrameskip-v4')) #, render_mode="human"))
  act = Tensor([[0]])

  obs, info = env.reset()
  img_0 = preprocess(obs)
  obs, reward, terminated, truncated, info = env.step(act.item())
  img_1 = preprocess(obs)

  print("testing worldmodel frame_cnn")
  test_x_emb = img_0.sequential(model.world_model.frame_cnn)
  real_x_emb = agent.world_model.frame_cnn(torch.Tensor(img_0.numpy()))
  assert test_x_emb.shape == real_x_emb.shape
  np.testing.assert_allclose(test_x_emb.numpy(), real_x_emb.numpy(), atol=1e-4)
  print("PASS")

  print("testing worldmodel act_emb")
  test_a_emb = model.world_model.act_emb(act)
  real_a_emb = agent.world_model.act_emb(torch.Tensor(act.numpy()).long())
  np.testing.assert_allclose(test_a_emb.numpy(), real_a_emb.numpy(), atol=1e-6)
  print("PASS")

  print("testing transformer")
  transformer_in = test_x_emb[0].cat(test_a_emb, dim=1)
  test_tout = model.world_model.transformer(transformer_in)
  real_tout = agent.world_model.transformer(torch.Tensor(transformer_in.numpy()))
  np.testing.assert_allclose(test_tout.numpy(), real_tout.numpy(), atol=1e-2)  # this atol might not be okay
  print("PASS")

  print("testing tokenizer")
  test_token = model.tokenizer(img_0, act, img_1)
  real_token = agent.tokenizer(torch.Tensor(img_0.numpy()), torch.Tensor(act.numpy()).long(), torch.Tensor(img_1.numpy()))
  np.testing.assert_allclose(test_token.q.numpy(), real_token.q.numpy(), atol=1e-6)
  np.testing.assert_allclose(test_token.tokens.numpy(), real_token.tokens.numpy(), atol=1e-6)
  print("PASS")

  print("testing tokenizer encode/decode")
  test_image = model.tokenizer.encode_decode(img_0, act, img_1)
  real_image = agent.tokenizer.encode_decode(torch.Tensor(img_0.numpy()), torch.Tensor(act.numpy()).long(), torch.Tensor(img_1.numpy()))
  np.testing.assert_allclose(test_image.numpy(), real_image.numpy(), atol=1e-6)  # one is a tiny bit off
  print("PASS")

  print("testing actor critic")
  model.actor_critic['model'](img_0)
  test_hx, test_cx = model.actor_critic['model'].hx, model.actor_critic['model'].cx
  model.actor_critic['model'](img_1)
  test_hx_2, test_cx_2 = model.actor_critic['model'].hx, model.actor_critic['model'].cx
  x = agent.actor_critic.model.cnn(torch.Tensor(img_0.numpy()))
  real_hx, real_cx = agent.actor_critic.model.lstm(x)
  x = agent.actor_critic.model.cnn(torch.Tensor(img_1.numpy()))
  real_hx_2, real_cx_2 = agent.actor_critic.model.lstm(x, (real_hx, real_cx))
  # NOTE: this is wrong in new torch?
  np.testing.assert_allclose(test_hx.numpy(), real_hx.numpy(), atol=1e-5)
  np.testing.assert_allclose(test_cx.numpy(), real_cx.numpy(), atol=1e-5)
  np.testing.assert_allclose(test_hx_2.numpy(), real_hx_2.numpy(), atol=1e-5)
  np.testing.assert_allclose(test_cx_2.numpy(), real_cx_2.numpy(), atol=1e-5)
  print("PASS")

if __name__ == "__main__":
  main()