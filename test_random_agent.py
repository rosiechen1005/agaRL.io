"""Random agent baseline: continuous or discrete action space with optional early stopping."""
import gym
import numpy as np
import gym_agario  # noqa: F401

env_id = "agario-grid-v0"
num_episodes = 10
max_steps = 1000
bad_reward_threshold = -10
bad_reward_limit = 1

# Use DiscretizedAgarioEnv for 363 discrete actions; otherwise continuous (move, 0).
# from discretized_agario_env_fixed import DiscretizedAgarioEnv
# env = DiscretizedAgarioEnv()
env = gym.make(env_id)

episode_lengths = []
episode_rewards = []
consec_bad_reward_steps = 0

for ep in range(num_episodes):
    out = env.reset()
    obs = out[0] if isinstance(out, (tuple, list)) and len(out) == 2 else out
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < max_steps:
        move = np.random.uniform(-1, 1, size=(2,)).astype(np.float32)
        disc = np.random.choice(3)
        action = (move, int(disc))
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, reward, done, info = step_out

        total_reward += reward
        steps += 1
        consec_bad_reward_steps = consec_bad_reward_steps + 1 if reward <= 0 else 0
        if consec_bad_reward_steps >= bad_reward_limit and total_reward <= bad_reward_threshold:
            break

    episode_lengths.append(steps)
    episode_rewards.append(total_reward)
    print(f"Episode {ep+1}: Steps={steps}, Total reward={total_reward:.2f}")

env.close()
print(f"\nSummary ({num_episodes} episodes): Avg steps={np.mean(episode_lengths):.1f}, Avg reward={np.mean(episode_rewards):.2f}")