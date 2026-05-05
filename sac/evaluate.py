import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deep_rl'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch as th

from deep_rl.sac_model import SoftActorCritic
from simulation.sim_env import SimEnv


def play(actor_path, max_steps=20000):
    env = SimEnv(human=True)
    obs_space = env.observation_space
    action_space = env.action_space

    model = SoftActorCritic(obs_space, action_space, [400, 300])
    model.actor.load_state_dict(th.load(actor_path, map_location='cpu', weights_only=True))
    model.eval()

    obs_low = obs_space.low.astype(np.float32)
    obs_scale = (obs_space.high - obs_space.low).astype(np.float32)

    def normalize(obs):
        return (obs - obs_low) / (obs_scale + 1e-8)

    observation, _ = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0
    info = {}

    while not done and step_count < max_steps:
        obs_norm = normalize(observation)
        action = model.get_deterministic_action(
            th.from_numpy(obs_norm)
        ).numpy()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

    print(
        f"steps: {step_count}",
        f"total_reward: {total_reward:.3f}",
        f"progress: {info.get('progress', 0):.3f}",
        f"finished_lap: {info.get('finished_lap', False)}",
    )
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('actor_path', help='Path to best_actor_N.pth')
    parser.add_argument('--max_steps', type=int, default=20000)
    args = parser.parse_args()
    play(args.actor_path, max_steps=args.max_steps)
