import gym
import numpy as np
import itertools
from gym import spaces


class DiscretizedAgarioEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.base_env = gym.make("agario-grid-v0")

        # Discretize x and y in [-5, ..., 5] and combine with action_type ∈ {0, 1, 2}
        self.motion_vals = list(range(-5, 6))  # 11 values
        self.action_types = [0, 1, 2]  # do nothing, split, feed
        self.actions = list(itertools.product(self.motion_vals, self.motion_vals, self.action_types))
        self.action_space = spaces.Discrete(len(self.actions))  # 11*11*3 = 363

        # CNN-compatible observation (shape set after first reset)
        self._obs_shape = None
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(128, 128, 10), dtype=np.float32)

    def reset(self):
        out = self.base_env.reset()
        obs = out[0] if isinstance(out, (tuple, list)) and len(out) == 2 else out
        obs = np.asarray(obs)
        if self._obs_shape is None:
            self._obs_shape = obs.shape
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=self._obs_shape, dtype=np.float32)
        return self._scale_obs(obs)

    def step(self, action_idx):
        dx, dy, act = self.actions[action_idx]
        direction = np.array([dx, dy], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm > 1:
            direction = direction / norm
        # Support both 4- and 5-tuple step returns (gym 0.26+)
        step_out = self.base_env.step((direction, act))
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out
        obs = np.asarray(obs)
        return self._scale_obs(obs), reward, done, info

    def _scale_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.max() > 1.0:
            obs = obs / 255.0
        return np.clip(obs, 0.0, 1.0)

    def render(self, mode="human"):
        self.base_env.render(mode=mode)

    def close(self):
        self.base_env.close()
