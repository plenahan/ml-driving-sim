import numpy as np
from gymnasium import Env
from controller import Controller
from replay_buffer import ReplayBuffer
from deep_rl import timed_decorator


class Runner:
    """
    Runner class responsible for managing the interactions between the controller and the environment.

    The Runner class is a helper class that encapsulates the running of an agent within an environment.
    It is initialized with an environment, a controller, and configuration parameters.
    It also maintains a buffer for storing environment states, actions taken, rewards received,
    next states, and done statuses during the interaction between the controller and the environment.

    Attributes:
        env (Environment): An instance of the environment for the agent to interact with.
        controller (Controller): The controller (or agent) that will interact with the environment.
        config_params (dict): A dictionary of configuration parameters.
        s (object): The current state of the environment.
        rewards (List[List[float]]): A list of lists, where each inner list contains the rewards for one episode.
        replay_buffer_shapes (dict): A dictionary where the keys are 'states', 'actions', 'rewards', 'next_states', and
                                     'dones', and the values are tuples indicating the shape of the corresponding data.
        info_buffer_shapes (dict): A dictionary where the keys are info keys and the values are tuples indicating the
                                   shape of the corresponding data.
    """
    def __init__(self, env: Env, controller: Controller, config_params: dict):
        """
        Initializes the Runner with the given environment, controller, and configuration parameters.

        Args:
            env (Environment): An instance of the environment for the agent to interact with.
            controller (Controller): The controller (or agent) that will interact with the environment.
            config_params (dict): A dictionary of configuration parameters.

        Raises:
            Warning: If the `info` returned from environment reset is not a dictionary.
        """
        self.env = env
        self.controller = controller
        self.config_params = config_params
        self.s = None
        self.rewards = [[]]
        self._obs_low = env.observation_space.low.astype(np.float32)
        self._obs_scale = (env.observation_space.high - env.observation_space.low).astype(np.float32)
        self.replay_buffer_shapes = {'states': self.env.observation_space.shape,
                                     'actions': self.env.action_space.shape,
                                     'rewards': (1,),
                                     'next_states': self.env.observation_space.shape,
                                     'dones': (1,)}

        # Set up the info buffer shapes (used for plotting returns).
        # Reset info may be empty, so probe one step to discover the actual structure.
        _, info = self.env.reset()
        if not (isinstance(info, dict) and len(info) > 0):
            probe_action = self.env.action_space.sample()
            _, _, _, _, info = self.env.step(probe_action)
            self.env.reset()
        if isinstance(info, dict) and len(info) > 0:
            self.info_buffer_shapes = {}
            for key, value in info.items():
                self.info_buffer_shapes[key] = value.shape if hasattr(value, 'shape') else (1,)
        else:
            self.info_buffer_shapes = None
            print('Warning: info is not a dict. Info buffer will not be used.')

    def _normalize_obs(self, obs):
        return (obs - self._obs_low) / (self._obs_scale + 1e-8)

    @timed_decorator
    def run_steps(self, num_steps):
        """
        Run the agent for a specified number of steps, storing the transitions in a replay buffer and tracking rewards.

        This method executes the agent's interactions with the environment for a given number of steps. During each step,
        the agent chooses an action based on its current state, performs the action, and stores the resulting state,
        action, reward, next state, and termination status in the replay buffer. It also tracks the rewards for each
        episode. If an episode ends before the specified number of steps is reached, the environment is reset and the
        agent continues until the total number of steps is completed.

        Args:
            num_steps (int): The number of steps to run the agent in the environment.

        Returns:
            memory (ReplayBuffer): A trimmed replay buffer containing the collected transitions.
            rewards (List[List[float]] or None): A list of lists, where each inner list contains the rewards for one
                episode. If no episode ends within the specified number of steps, returns None.
            episode_lengths (List[int] or None): A list of integers representing the lengths of the episodes that ended
                during the specified number of steps. If no episode ends within the specified number of steps, returns
                None.
        """
        memory = ReplayBuffer(num_steps, self.replay_buffer_shapes)
        info_memory = ReplayBuffer(num_steps, self.info_buffer_shapes)
        self.rewards = [self.rewards[-1]]
        if self.s is None:
            self.s, _ = self.env.reset()
            self.s = self._normalize_obs(self.s)

        for step in range(num_steps):
            a = self.controller.choose(self.s)
            ns, r, t, d, i = self.env.step(a)
            ns = self._normalize_obs(ns)
            memory.store(states=self.s, actions=a, rewards=r, next_states=ns, dones=d or t)
            if self.info_buffer_shapes is not None:
                info_memory.store(**i)
            self.s = ns
            self.rewards[-1].append(r)
            if d or t or len(self.rewards[-1]) == self.config_params['max_episode_steps'] - 1:
                self.s, _ = self.env.reset()
                self.s = self._normalize_obs(self.s)
                self.rewards.append([])
        if len(self.rewards) > 1:
            return memory.trim(), self.rewards[:-1], [len(epi_r) for epi_r in self.rewards[:-1]], info_memory.trim()
        return memory.trim(), None, None, info_memory.trim()
