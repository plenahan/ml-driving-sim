import torch
from network import FeedForwardNN
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch import nn
import numpy as np
import os

class PPO:
    def __init__(self, env):
        self._init_hyperparameters()
        self.env = env
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.actor = FeedForwardNN(self.observation_space, self.action_space)
        self.critic = FeedForwardNN(self.observation_space, 1)
        self.log_std = nn.Parameter(torch.full((self.action_space,), self.initial_log_std, dtype=torch.float32))
        self.actor_optimizer = Adam(list(self.actor.parameters()) + [self.log_std], lr=self.learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.learning_rate)

    def _init_hyperparameters(self):
        self.iterations_per_batch = 16384
        self.max_iterations_per_episode = 4096
        self.gamma = 0.99
        self.n_updates_per_iteration = 6
        self.clip = 0.2
        self.learning_rate = 2.5e-4
        self.entropy_coefficient_start = 0.02
        self.entropy_coefficient_end = 0.001
        self.entropy_decay_steps = 1_000_000
        self.initial_log_std = -0.5

    def _entropy_coefficient(self, total_steps):
        progress = min(1.0, total_steps / self.entropy_decay_steps)
        return self.entropy_coefficient_start + progress * (self.entropy_coefficient_end - self.entropy_coefficient_start)

    def _distribution(self, mean):
        std = torch.exp(self.log_std)
        covariance_matrix = torch.diag(std * std)
        return MultivariateNormal(mean, covariance_matrix)

    def learn(self, num_iterations):
        t_so_far = 0
        while t_so_far < num_iterations:
            batch_observations, batch_actions, batch_log_probabilities, batch_rewards_to_go, batch_episode_lengths, batch_episode_returns = self.rollout()
            t_so_far += np.sum(batch_episode_lengths)
            entropy_coefficient = self._entropy_coefficient(t_so_far)
            v, _, _ = self.evaluate(batch_observations, batch_actions)
            advantage = batch_rewards_to_go - v.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                v, current_log_probabilities, entropy = self.evaluate(batch_observations, batch_actions)

                ratio = torch.exp(current_log_probabilities - batch_log_probabilities)

                surrogate1 = ratio * advantage
                surrogate2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage

                actor_loss = (-torch.min(surrogate1, surrogate2)).mean() - entropy_coefficient * entropy.mean()
                critic_loss = torch.nn.MSELoss()(v, batch_rewards_to_go)

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
            print(
                "t_so_far:",
                int(t_so_far),
                "avg_ep_len:",
                float(np.mean(batch_episode_lengths)),
                "avg_ep_return:",
                float(np.mean(batch_episode_returns)),
                "entropy_coef:",
                float(entropy_coefficient),
                "action_std:",
                float(torch.exp(self.log_std).mean().detach()),
            )

    def rollout(self):
        batch_observations = []
        batch_actions = []
        batch_log_probabilities = []
        batch_rewards = []
        batch_rewards_to_go = []
        batch_episode_lengths = []
        batch_episode_returns = []

        t_so_far = 0

        while t_so_far < self.iterations_per_batch:
            episode_rewards = []
            episode_return = 0.0
            episode_length = 0
            observation = self.env.reset()
            done = False
            for episode_i in range(self.max_iterations_per_episode):
                t_so_far += 1
                episode_length += 1
                batch_observations.append(observation)
                
                action, log_probability = self.get_action(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)

                episode_return += reward
                done = terminated or truncated
                
                episode_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probabilities.append(log_probability.item())

                if done:
                    break

            batch_rewards.append(episode_rewards)
            batch_episode_returns.append(episode_return)
            batch_episode_lengths.append(episode_length)

        batch_observations = torch.tensor(np.array(batch_observations), dtype=torch.float32)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float32)
        batch_log_probabilities = torch.tensor(np.array(batch_log_probabilities), dtype=torch.float32)

        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)
        # batch_episode_returns = torch.tensor(batch_episode_returns, dtype=torch.float32)

        return batch_observations, batch_actions, batch_log_probabilities, batch_rewards_to_go, batch_episode_lengths, batch_episode_returns

    def get_action(self, observation):
        mean = self.actor(observation)
        distribution = self._distribution(mean)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)

        return action.detach().numpy(), log_probability.detach()

    def predict(self, observation):
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(observation_tensor)

        action_np = action.detach().numpy()
        return np.clip(action_np, self.env.action_space.low, self.env.action_space.high)

    def compute_rewards_to_go(self, batch_rewards):
        batch_rewards_to_go = []
        for episode_rewards in reversed(batch_rewards):
            discounted_reward = 0

            for reward in reversed(episode_rewards):
                discounted_reward = reward + self.gamma * discounted_reward
                batch_rewards_to_go.insert(0, discounted_reward)

        print("Sum of Batch rewards to go:", sum(batch_rewards_to_go))

        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float32)

        return batch_rewards_to_go

    def evaluate(self, batch_observations, batch_actions):
        mean = self.actor(batch_observations)
        distribution = self._distribution(mean)
        log_probabilities = distribution.log_prob(batch_actions)
        entropy = distribution.entropy()
        return self.critic(batch_observations).squeeze(), log_probabilities, entropy


    def save(self, actor_path, critic_path):
        actor_dir = os.path.dirname(actor_path)
        critic_dir = os.path.dirname(critic_path)
        if actor_dir:
            os.makedirs(actor_dir, exist_ok=True)
        if critic_dir:
            os.makedirs(critic_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.actor.state_dict(),
                "log_std": self.log_std.detach().cpu(),
            },
            actor_path,
        )
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, actor_path, critic_path=None):
        actor_state = torch.load(actor_path, map_location="cpu")
        if isinstance(actor_state, dict) and "model_state_dict" in actor_state:
            self.actor.load_state_dict(actor_state["model_state_dict"])
            loaded_log_std = actor_state.get("log_std")
            if loaded_log_std is not None:
                with torch.no_grad():
                    self.log_std.copy_(loaded_log_std)
        else:
            # Backward compatibility with older checkpoints that only saved actor weights.
            self.actor.load_state_dict(actor_state)
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
