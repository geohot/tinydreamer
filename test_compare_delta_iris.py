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

  print("testing tokenizer")
  test_token = model.tokenizer(img_0, act, img_1)
  real_token = agent.tokenizer(torch.Tensor(img_0.numpy()), torch.Tensor(act.numpy()).long(), torch.Tensor(img_1.numpy()))
  np.testing.assert_allclose(test_token.q.numpy(), real_token.q.numpy(), atol=1e-6)
  np.testing.assert_allclose(test_token.tokens.numpy(), real_token.tokens.numpy(), atol=1e-6)
  print("PASS")

  print("testing tokenizer encode/decode")
  test_image = model.tokenizer.encode_decode(img_0, act, img_1)
  real_image = agent.tokenizer.encode_decode(torch.Tensor(img_0.numpy()), torch.Tensor(act.numpy()).long(), torch.Tensor(img_1.numpy()))
  np.testing.assert_allclose(test_image.numpy(), real_image.numpy(), atol=1e-2)  # one is a tiny bit off
  print("PASS")


if __name__ == "__main__":
  main()