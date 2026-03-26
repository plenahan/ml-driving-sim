import torch
import os
from network import FeedForwardNN
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import numpy as np


class PPO:
    def __init__(self, env):
        self._init_hyperparameters()
        self.env = env
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]

        self.actor = FeedForwardNN(self.observation_space, self.action_space)
        self.critic = FeedForwardNN(self.observation_space, 1)

        self.action_std = torch.tensor(self.action_std, dtype=torch.float32)
        self.covariance_matrix = torch.diag(self.action_std * self.action_std)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_learning_rate)

        self.observation_scale = torch.tensor(
            np.maximum(self.env.observation_space.high, 1e-6), dtype=torch.float32
        )

    def _init_hyperparameters(self):
        self.iterations_per_batch = 4096
        self.max_iterations_per_episode = 1200
        self.gamma = 0.99
        self.gae_lambda = 0.95

        self.n_updates_per_iteration = 8
        self.minibatch_size = 256
        self.clip = 0.2
        self.target_kl = 0.015

        self.actor_learning_rate = 3e-4
        self.critic_learning_rate = 1e-3
        self.entropy_coefficient = 0.01
        self.max_grad_norm = 0.5

        self.action_std = [0.15, 0.15, 0.2]

    def learn(self, num_iterations):
        t_so_far = 0
        while t_so_far < num_iterations:
            (
                batch_observations,
                batch_actions,
                batch_log_probabilities,
                batch_returns,
                batch_advantages,
                batch_episode_lengths,
                batch_episode_returns,
                batch_episode_progresses,
            ) = self.rollout()

            t_so_far += np.sum(batch_episode_lengths)

            num_samples = batch_observations.shape[0]
            early_stop = False

            for _ in range(self.n_updates_per_iteration):
                indices = np.arange(num_samples)
                np.random.shuffle(indices)

                for start in range(0, num_samples, self.minibatch_size):
                    mb_idx = indices[start:start + self.minibatch_size]

                    mb_observations = batch_observations[mb_idx]
                    mb_actions = batch_actions[mb_idx]
                    mb_old_log_probabilities = batch_log_probabilities[mb_idx]
                    mb_returns = batch_returns[mb_idx]
                    mb_advantages = batch_advantages[mb_idx]

                    values, current_log_probabilities, entropy = self.evaluate(mb_observations, mb_actions)

                    ratio = torch.exp(current_log_probabilities - mb_old_log_probabilities)
                    surrogate1 = ratio * mb_advantages
                    surrogate2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_advantages

                    actor_loss = -torch.min(surrogate1, surrogate2).mean() - self.entropy_coefficient * entropy.mean()
                    critic_loss = torch.nn.MSELoss()(values, mb_returns)

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()

                    approx_kl = (mb_old_log_probabilities - current_log_probabilities).mean().item()
                    if approx_kl > self.target_kl:
                        early_stop = True
                        break

                if early_stop:
                    break

            print(
                "t_so_far:",
                int(t_so_far),
                "avg_ep_len:",
                float(np.mean(batch_episode_lengths)),
                "avg_ep_return:",
                float(np.mean(batch_episode_returns)),
                "avg_ep_progress:",
                float(np.mean(batch_episode_progresses)),
            )

    def rollout(self):
        batch_observations = []
        batch_actions = []
        batch_log_probabilities = []
        batch_returns = []
        batch_advantages = []
        batch_episode_lengths = []
        batch_episode_returns = []
        batch_episode_progresses = []

        t_so_far = 0

        while t_so_far < self.iterations_per_batch:
            episode_rewards = []
            episode_values = []
            episode_dones = []

            episode_return = 0.0
            observation, _ = self.env.reset()
            done = False
            info = {}
            episode_i = -1

            for episode_i in range(self.max_iterations_per_episode):
                t_so_far += 1
                batch_observations.append(observation)

                observation_tensor = torch.tensor(observation, dtype=torch.float32)
                normalized_observation = self._normalize_observation(observation_tensor)
                value = self.critic(normalized_observation).item()

                action, log_probability = self.get_action(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)

                done = terminated or truncated
                episode_return += reward

                episode_rewards.append(reward)
                episode_values.append(value)
                episode_dones.append(done)

                batch_actions.append(action)
                batch_log_probabilities.append(log_probability)

                if done or t_so_far >= self.iterations_per_batch:
                    break

            episode_advantages, episode_returns = self.compute_gae(episode_rewards, episode_values, episode_dones)
            batch_advantages.extend(episode_advantages)
            batch_returns.extend(episode_returns)
            batch_episode_lengths.append(episode_i + 1)
            batch_episode_returns.append(episode_return)
            batch_episode_progresses.append(float(info.get("progress", 0.0)))

        batch_observations = torch.tensor(np.array(batch_observations), dtype=torch.float32)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float32)
        batch_log_probabilities = torch.tensor(np.array(batch_log_probabilities), dtype=torch.float32)
        batch_returns = torch.tensor(np.array(batch_returns), dtype=torch.float32)
        batch_advantages = torch.tensor(np.array(batch_advantages), dtype=torch.float32)

        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-10)

        return (
            batch_observations,
            batch_actions,
            batch_log_probabilities,
            batch_returns,
            batch_advantages,
            batch_episode_lengths,
            batch_episode_returns,
            batch_episode_progresses,
        )

    def _normalize_observation(self, observation_tensor):
        return observation_tensor / self.observation_scale

    def _policy_mean(self, observations):
        raw_output = self.actor(observations)
        throttle = torch.sigmoid(raw_output[..., 0:1])
        brake = torch.sigmoid(raw_output[..., 1:2])
        steering = torch.tanh(raw_output[..., 2:3])
        return torch.cat([throttle, brake, steering], dim=-1)

    def _distribution(self, observations):
        normalized_observations = self._normalize_observation(observations)
        mean = self._policy_mean(normalized_observations)
        return MultivariateNormal(mean, self.covariance_matrix)

    def get_action(self, observation):
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        distribution = self._distribution(observation_tensor)

        with torch.no_grad():
            action = distribution.sample()
            log_probability = distribution.log_prob(action)

        action_np = action.detach().numpy()
        action_np = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)
        return action_np, float(log_probability.item())

    def predict(self, observation, deterministic=True):
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        distribution = self._distribution(observation_tensor)

        with torch.no_grad():
            if deterministic:
                action = distribution.mean
            else:
                action = distribution.sample()

        action_np = action.detach().numpy()
        return np.clip(action_np, self.env.action_space.low, self.env.action_space.high)

    def compute_gae(self, rewards, values, dones):
        advantages = []
        returns = []
        gae = 0.0
        next_value = 0.0

        for i in reversed(range(len(rewards))):
            not_done = 1.0 - float(dones[i])
            delta = rewards[i] + self.gamma * next_value * not_done - values[i]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_value = values[i]

        return advantages, returns

    def evaluate(self, batch_observations, batch_actions):
        distribution = self._distribution(batch_observations)
        log_probabilities = distribution.log_prob(batch_actions)
        entropy = distribution.entropy()

        normalized_observations = self._normalize_observation(batch_observations)
        values = self.critic(normalized_observations).squeeze()
        return values, log_probabilities, entropy

    def save(self, actor_path, critic_path):
        actor_dir = os.path.dirname(actor_path)
        critic_dir = os.path.dirname(critic_path)
        if actor_dir:
            os.makedirs(actor_dir, exist_ok=True)
        if critic_dir:
            os.makedirs(critic_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, actor_path, critic_path=None):
        self.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))

