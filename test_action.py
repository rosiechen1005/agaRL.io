"""Run random actions on agario-grid-v0 and report episode stats."""
import gym
import numpy as np
import gym_agario  # noqa: F401

env = gym.make("agario-grid-v0")
num_episodes = 50
episode_lengths = []
episode_rewards = []

for i in range(num_episodes):
    out = env.reset()
    obs = out[0] if isinstance(out, (tuple, list)) and len(out) == 2 else out
    done = False
    total_reward = 0
    steps = 0

    while not done:
        action = env.action_space.sample()
        # Clip continuous part to [-1, 1] for base env
        if isinstance(action, (tuple, list)):
            move, disc = action[0], action[1]
            move = np.clip(np.asarray(move, dtype=np.float32), -1.0, 1.0)
            action = (move, disc)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out
        total_reward += reward
        steps += 1

    episode_lengths.append(steps)
    episode_rewards.append(total_reward)
    print(f"Episode {i+1}: length={steps}, total_reward={total_reward:.2f}")

print("\n--- Summary ---")
print(f"Average Episode Length: {sum(episode_lengths) / num_episodes:.1f}")
print(f"Average Total Reward: {sum(episode_rewards) / num_episodes:.2f}")
env.close()