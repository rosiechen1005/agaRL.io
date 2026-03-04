"""
Train a PPO agent to play Agar.io (AgarLE).

Goal: train the agent to survive as long as possible without explicitly telling it
how the game works.
- Get larger by eating pellets or smaller cells.
- Get smaller by hitting a virus or getting eaten.

Within each round:
- At each timestep, the policy (actor) chooses a direction to move in.
- We observe the reward (change in size) and step the environment.
- We observe changes in the surrounding environment (grid arrays: player cell,
  pellets, viruses, other cells). This is how we train the policy + value networks.

PPO: relatively simple yet powerful. Two neural networks:
- Policy network (actor): chooses actions based on observed surroundings / learned strategies.
- Value network (critic): calculates how advantageous an action was based on gameplay data.
PPO restricts how much the policy can change at once.
"""

import argparse
import gym
import numpy as np
import gym_agario  # noqa: F401  # register envs

from ppo.agent import PPOAgent


def get_obs_shape(env):
    space = env.observation_space
    if hasattr(space, "shape"):
        return tuple(space.shape)
    return (128, 128, 10)  # default grid


def run_training(
    env_id="agario-grid-v0",
    total_timesteps=100_000,
    rollout_steps=512,
    n_epochs=4,
    batch_size=64,
    lr=3e-4,
    gamma=0.99,
    seed=0,
    save_path=None,
    log_interval=10,
    **env_kwargs,
):
    env = gym.make(env_id, **env_kwargs)
    if seed is not None:
        env.seed(seed)
        np.random.seed(seed)

    obs_shape = get_obs_shape(env)
    agent = PPOAgent(obs_shape, lr=lr, gamma=gamma)

    global_step = 0
    episode_rewards = []
    episode_lengths = []
    current_ep_reward = 0
    current_ep_length = 0

    _reset = env.reset()
    obs = _reset[0] if isinstance(_reset, (list, tuple)) and len(_reset) == 2 else _reset
    obs = np.asarray(obs)

    while global_step < total_timesteps:
        for _ in range(rollout_steps):
            move, disc, log_prob, value = agent.select_action(obs, deterministic=False)
            action = (np.array(move, dtype=np.float32), int(disc))
            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_out
            next_obs = np.asarray(next_obs)

            agent.store_step(obs, move, disc, reward, float(done), log_prob, value)
            current_ep_reward += reward
            current_ep_length += 1
            obs = next_obs
            global_step += 1

            if done:
                episode_rewards.append(current_ep_reward)
                episode_lengths.append(current_ep_length)
                _reset = env.reset()
                obs = _reset[0] if isinstance(_reset, (list, tuple)) and len(_reset) == 2 else _reset
                obs = np.asarray(obs)
                current_ep_reward = 0
                current_ep_length = 0

        last_done = float(done)  # from last step of rollout
        metrics = agent.update(obs, last_done, n_epochs=n_epochs, batch_size=batch_size)

        if log_interval and (global_step // rollout_steps) % log_interval == 0 and episode_rewards:
            mean_rew = np.mean(episode_rewards[-50:])
            mean_len = np.mean(episode_lengths[-50:])
            print(
                f"step {global_step} | reward {mean_rew:.1f} | length {mean_len:.0f} | "
                f"policy_loss {metrics['policy_loss']:.4f} | value_loss {metrics['value_loss']:.4f}"
            )

    env.close()
    if save_path:
        import torch
        torch.save(agent.model.state_dict(), save_path)
        print(f"Saved model to {save_path}")
    return agent, episode_rewards, episode_lengths


def main():
    parser = argparse.ArgumentParser(description="Train PPO on AgarLE")
    parser.add_argument("--env", default="agario-grid-v0")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default=None, help="Path to save model")
    parser.add_argument("--difficulty", default="normal", choices=["normal", "empty", "trivial"])
    args = parser.parse_args()

    run_training(
        env_id=args.env,
        total_timesteps=args.timesteps,
        rollout_steps=args.rollout_steps,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        save_path=args.save,
        difficulty=args.difficulty,
    )


if __name__ == "__main__":
    main()
