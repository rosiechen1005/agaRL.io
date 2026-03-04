"""
PPO agent: relatively simple yet powerful.

- Policy network (actor) chooses actions based on observed surroundings / learned strategies.
- Value network (critic) calculates how advantageous an action was based on gameplay data.
- PPO restricts how much the policy can change at once (clipped objective).

At each timestep the agent: chooses direction to move, then observes reward (change in size)
and changes in the surrounding environment (grid arrays: player, pellets, viruses, others).
"""

import numpy as np
import torch
import torch.nn.functional as F

from .networks import ActorCritic


class RolloutBuffer:
    """Stores one rollout for PPO (observations, actions, rewards, etc.)."""

    def __init__(self, obs_shape, move_dim=2, device="cpu"):
        self.obs_shape = obs_shape
        self.move_dim = move_dim
        self.device = device
        self.obs = []
        self.move = []
        self.disc = []
        self.reward = []
        self.done = []
        self.log_prob = []
        self.value = []

    def add(self, obs, move, disc, reward, done, log_prob, value):
        self.obs.append(obs)
        self.move.append(move)
        self.disc.append(disc)
        self.reward.append(reward)
        self.done.append(done)
        self.log_prob.append(log_prob)
        self.value.append(value)

    def compute_gae(self, last_value, last_done, gamma=0.99, lam=0.95):
        """Compute generalized advantage estimation."""
        rewards = np.array(self.reward, dtype=np.float32)
        dones = np.array(self.done, dtype=np.float32)
        values = np.array(self.value + [last_value], dtype=np.float32)
        gae = 0
        advantages = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * gae * (1 - dones[t])
            advantages.append(gae)
        advantages = np.array(advantages[::-1], dtype=np.float32)
        returns = advantages + np.array(self.value, dtype=np.float32)
        return advantages, returns

    def get_batches(self, batch_size, advantages, returns):
        """Yield random mini-batches for PPO update."""
        n = len(self.obs)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            obs_b = torch.from_numpy(np.stack([self.obs[i] for i in idx])).to(self.device)
            move_b = torch.from_numpy(np.stack([self.move[i] for i in idx])).to(self.device)
            disc_b = torch.from_numpy(np.stack([self.disc[i] for i in idx])).long().to(self.device)
            log_prob_old = torch.from_numpy(np.array([self.log_prob[i] for i in idx])).to(self.device)
            adv_b = torch.from_numpy(advantages[idx]).to(self.device)
            ret_b = torch.from_numpy(returns[idx]).to(self.device)
            yield obs_b, move_b, disc_b, log_prob_old, adv_b, ret_b

    def clear(self):
        self.obs.clear()
        self.move.clear()
        self.disc.clear()
        self.reward.clear()
        self.done.clear()
        self.log_prob.clear()
        self.value.clear()


class PPOAgent:
    """
    PPO agent: collect experience, then update policy and value networks.
    Goal: train agent to survive as long as possible; reward = change in size.
    """

    def __init__(
        self,
        obs_shape,
        device=None,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(obs_shape).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.buffer = RolloutBuffer(obs_shape, move_dim=2, device=self.device)

    def select_action(self, obs, deterministic=False):
        """Choose direction to move and discrete action from current observation."""
        with torch.no_grad():
            obs_t = self._to_tensor(obs)
            move, disc, log_prob, value, *_ = self.model.get_action_and_log_prob(
                obs_t, deterministic=deterministic
            )
            move_np = move.cpu().numpy().squeeze()
            disc_np = disc.cpu().item() if disc.dim() == 0 else disc.cpu().numpy().squeeze()
            log_prob_np = log_prob.cpu().item() if log_prob.dim() == 0 else log_prob.cpu().numpy().squeeze()
            value_np = value.cpu().item() if value.dim() == 0 else value.cpu().numpy().squeeze()
        return move_np, disc_np, log_prob_np, value_np

    def _to_tensor(self, obs):
        if isinstance(obs, np.ndarray):
            t = torch.from_numpy(obs).float()
        else:
            t = torch.as_tensor(obs, dtype=torch.float32)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t.to(self.device)

    def store_step(self, obs, move, disc, reward, done, log_prob, value):
        """Record one timestep for later PPO update."""
        self.buffer.add(
            obs.copy() if isinstance(obs, np.ndarray) else obs,
            np.array(move, dtype=np.float32),
            int(disc),
            float(reward),
            float(done),
            float(log_prob),
            float(value),
        )

    def update(self, last_obs, last_done, n_epochs=4, batch_size=64):
        """
        Update policy and value networks using collected rollout.
        PPO restricts how much the policy can change at once (clipped objective).
        """
        with torch.no_grad():
            last_obs_t = self._to_tensor(last_obs)
            _, _, _, last_value, *_ = self.model.get_action_and_log_prob(last_obs_t, deterministic=True)
            last_value = last_value.cpu().item()
        advantages, returns = self.buffer.compute_gae(
            last_value, last_done, gamma=self.gamma, lam=self.gae_lambda
        )
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_batches = 0
        for _ in range(n_epochs):
            for obs_b, move_b, disc_b, log_prob_old, adv_b, ret_b in self.buffer.get_batches(
                batch_size, advantages, returns
            ):
                log_prob, value, entropy = self.model.evaluate_actions(obs_b, move_b, disc_b)
                ratio = (log_prob - log_prob_old).exp()
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, ret_b)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_batches += 1
        self.buffer.clear()
        return {
            "policy_loss": total_policy_loss / max(n_batches, 1),
            "value_loss": total_value_loss / max(n_batches, 1),
            "entropy": total_entropy / max(n_batches, 1),
        }
