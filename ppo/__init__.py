# PPO agent for AgarLE: policy (actor) + value (critic) networks.
# Goal: train agent to survive as long as possible from rewards (change in size).
# Observation: grid arrays (player cell, pellets, viruses, other cells).

from .networks import ActorCritic
from .agent import PPOAgent

__all__ = ["ActorCritic", "PPOAgent"]
