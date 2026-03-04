"""
PPO networks for AgarLE.

- Policy network (actor): chooses direction to move and discrete action (split/feed/none)
  based on observed surroundings (grid arrays: player, pellets, viruses, other cells).
- Value network (critic): estimates how advantageous the current state is (for GAE).

Environment is represented with arrays; the CNN processes these spatial observations.
"""

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """Shared CNN that processes grid observation (H, W, C) into a feature vector."""

    def __init__(self, obs_shape, out_dim=256, channel_dims=(32, 64, 64)):
        super().__init__()
        # obs_shape: (H, W, C) e.g. (128, 128, 10)
        h, w, c = obs_shape
        layers = []
        in_c = c
        for ch in channel_dims:
            layers += [
                nn.Conv2d(in_c, ch, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_c = ch
        self.conv = nn.Sequential(*layers)
        # approximate spatial size after 3 strides of 2: 128 -> 64 -> 32 -> 16
        self._h, self._w = h // (2 ** len(channel_dims)), w // (2 ** len(channel_dims))
        self._out_dim = out_dim
        self.fc = nn.Sequential(
            nn.Linear(in_c * self._h * self._w, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (N, H, W, C) -> (N, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] in (10, 4, 3):  # channels last
            x = x.permute(0, 3, 1, 2)
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        if x.max() > 1.0:
            x = x / max(float(x.max()), 1.0)
        feat = self.conv(x)
        feat = feat.reshape(feat.size(0), -1)
        return self.fc(feat)


class ActorCritic(nn.Module):
    """
    Two networks in one module (shared backbone):
    - Policy (actor): chooses action (direction + discrete) from observed surroundings.
    - Value (critic): estimates how advantageous the current state is.
    """

    def __init__(
        self,
        obs_shape,
        move_dim=2,
        discrete_actions=3,
        hidden_dim=256,
        log_std_init=-0.5,
    ):
        super().__init__()
        self.encoder = ConvEncoder(obs_shape, out_dim=hidden_dim)
        self.move_dim = move_dim
        self.discrete_actions = discrete_actions

        # Actor heads: direction (Gaussian) + discrete (categorical)
        self.actor_move_mean = nn.Linear(hidden_dim, move_dim)
        self.actor_move_log_std = nn.Parameter(torch.full((move_dim,), log_std_init))
        self.actor_discrete = nn.Linear(hidden_dim, discrete_actions)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        feat = self.encoder(obs)
        move_mean = torch.tanh(self.actor_move_mean(feat))  # [-1, 1]
        move_log_std = self.actor_move_log_std.clamp(-20, 2).expand_as(move_mean)
        discrete_logits = self.actor_discrete(feat)
        value = self.critic(feat).squeeze(-1)
        return move_mean, move_log_std, discrete_logits, value

    def get_action_and_log_prob(self, obs, deterministic=False):
        move_mean, move_log_std, discrete_logits, value = self.forward(obs)
        std = move_log_std.exp()
        dist_move = torch.distributions.Normal(move_mean, std)
        dist_disc = torch.distributions.Categorical(logits=discrete_logits)
        if deterministic:
            move = move_mean
            disc = discrete_logits.argmax(dim=-1)
        else:
            move = dist_move.sample()
            disc = dist_disc.sample()
        move = torch.clamp(move, -1.0, 1.0)
        log_prob_move = dist_move.log_prob(move).sum(dim=-1)
        log_prob_disc = dist_disc.log_prob(disc)
        log_prob = log_prob_move + log_prob_disc
        return (
            move,
            disc,
            log_prob,
            value,
            move_mean,
            move_log_std,
            discrete_logits,
        )

    def evaluate_actions(self, obs, move, disc):
        move_mean, move_log_std, discrete_logits, value = self.forward(obs)
        std = move_log_std.exp()
        dist_move = torch.distributions.Normal(move_mean, std)
        dist_disc = torch.distributions.Categorical(logits=discrete_logits)
        log_prob_move = dist_move.log_prob(move).sum(dim=-1)
        log_prob_disc = dist_disc.log_prob(disc)
        log_prob = log_prob_move + log_prob_disc
        entropy = dist_move.entropy().sum(dim=-1) + dist_disc.entropy()
        return log_prob, value, entropy
