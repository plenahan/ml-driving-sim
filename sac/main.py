import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deep_rl'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from deep_rl.experiment import SACExperiment, MultiExperiment
from deep_rl.sac_model import SoftActorCritic
from simulation.sim_env import SimEnv

SAC_CONFIG_PARAMS_LUNAR = {
    "max_steps": int(2E6),
    "replay_buffer_size": int(5E5),  # ~25 full episodes worth of experience
    "warmup_steps": int(1E4),
    "batch_size": 256,               # standard SAC batch size
    "max_episode_steps": -1,
    "runner_steps": 1000,            # collect a meaningful chunk per update cycle
    "gamma": 0.99,
    "grad_repeats": 200,              # 0.2 gradient steps per collected transition
    "alpha_lr": 1e-4,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha": 0.2,                    # standard continuous-control entropy coeff
    "tau": 5e-3,
    "render_graphs": False,
    "render_every": 5000,
    "print_every": 5000,
    "save_path": "./data/experiments_data/sac_simenv/",
    "render": True,
    "hidden_size": [400, 300],
    "device": 'cpu'
}

SAC_CONFIG_PARAMS_PENDULUM = {
    "max_steps": 50000,
    "replay_buffer_size": 5000,
    "warmup_steps": int(1E4),
    "batch_size": 200,
    "max_episode_steps": 400,
    "runner_steps": 200,
    "gamma": 0.9,
    "grad_repeats": 32,
    "alpha_lr": 0.001,
    "actor_lr": 0.001,
    "critic_lr": 0.002,
    "alpha": 1,
    "tau": 0.01,
    "render_every": 1,
    "print_every": 1,
    "save_path": "./experiments_data/sac_pendulum/",
    "render": True,
    "hidden_size": [32, 32],
    "device": 'cpu'
}

if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=np.nan)

    env = SimEnv(human=False)

    obv_space = env.observation_space
    action_space = env.action_space
    SAC_Constructor = (obv_space, action_space, SAC_CONFIG_PARAMS_LUNAR['hidden_size'])

    print("Model constructor: ", SAC_Constructor)
    multi_experiment = MultiExperiment(1, env, SACExperiment, SoftActorCritic, SAC_Constructor, SAC_CONFIG_PARAMS_LUNAR)
    multi_experiment.run_experiments()
