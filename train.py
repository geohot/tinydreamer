#!/usr/bin/env python3
import gymnasium as gym

if __name__ == "__main__":
  env = gym.make('ALE/Pong-v5', render_mode="human")  # remove render_mode in training
  obs, info = env.reset()
  episode_over = False
  while not episode_over:
    #action = policy(obs)  # to implement - use `env.action_space.sample()` for a random policy
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
  env.close()