import torch
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
        self.covariance = torch.full(size=(self.action_space,), fill_value=0.5)
        self.covariance_matrix = torch.diag(self.covariance)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.learning_rate)

    def _init_hyperparameters(self):
        self.iterations_per_batch = 1024
        self.max_iterations_per_episode = 1000
        self.gamma = 0.999
        self.n_updates_per_iteration = 4
        self.clip = 0.2
        self.learning_rate = 2.5e-4

    def learn(self, num_iterations):
        t_so_far = 0
        while t_so_far < num_iterations:
            batch_observations, batch_actions, batch_log_probabilities, batch_rewards_to_go, batch_episode_lengths = self.rollout()
            t_so_far += np.sum(batch_episode_lengths)
            print("t_so_far:", t_so_far)
            v, _ = self.evaluate(batch_observations, batch_actions)
            advantage = batch_rewards_to_go - v.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                v, current_log_probabilities = self.evaluate(batch_observations, batch_actions)

                ratio = torch.exp(current_log_probabilities - batch_log_probabilities)

                surrogate1 = ratio * advantage
                surrogate2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage

                actor_loss = (-torch.min(surrogate1, surrogate2)).mean()
                critic_loss = torch.nn.MSELoss()(v, batch_rewards_to_go)

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

    def rollout(self):
        batch_observations = []
        batch_actions = []
        batch_log_probabilities = []
        batch_rewards = []
        batch_rewards_to_go = []
        batch_episode_lengths = []

        t_so_far = 0

        while t_so_far < self.iterations_per_batch:
            episode_rewards = []
            observation, _ = self.env.reset()
            done = False
            for episode_i in range(self.max_iterations_per_episode):
                t_so_far += 1
                batch_observations.append(observation)
                
                action, log_probability = self.get_action(observation)
                observation, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated
                
                episode_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probabilities.append(log_probability)

                if done:
                    break

            batch_rewards.append(episode_rewards)
            batch_episode_lengths.append(episode_i + 1)

        batch_observations = torch.tensor(np.array(batch_observations), dtype=torch.float32)
        batch_actions = torch.tensor(np.array(batch_actions), dtype=torch.float32)
        batch_log_probabilities = torch.tensor(np.array(batch_log_probabilities), dtype=torch.float32)

        batch_rewards_to_go = self.compute_rewards_to_go(batch_rewards)

        return batch_observations, batch_actions, batch_log_probabilities, batch_rewards_to_go, batch_episode_lengths

    def get_action(self, observation):
        mean = self.actor(observation)
        distribution = MultivariateNormal(mean, self.covariance_matrix)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)

        return action.detach().numpy(), log_probability.detach()

    def compute_rewards_to_go(self, batch_rewards):
        batch_rewards_to_go = []
        for episode_rewards in reversed(batch_rewards):
            discounted_reward = 0

            for reward in episode_rewards:
                discounted_reward = reward + self.gamma * discounted_reward
                batch_rewards_to_go.insert(0, discounted_reward)

        print("Sum of Batch rewards to go:", sum(batch_rewards_to_go))

        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float32)

        return batch_rewards_to_go

    def evaluate(self, batch_observations, batch_actions):
        mean = self.actor(batch_observations)
        distribution = MultivariateNormal(mean, self.covariance_matrix)
        log_probabilities = distribution.log_prob(batch_actions)
        return self.critic(batch_observations).squeeze(), log_probabilities

