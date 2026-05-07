import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from network import FeedForwardNN
from torch.distributions import Normal, Independent
from torch.optim import Adam
import numpy as np


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count


class PPO:
    def __init__(self, env):
        self._init_hyperparameters()
        self.env = env
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]

        self.actor = FeedForwardNN(self.observation_space, self.action_space, output_gain=0.01)
        self.critic = FeedForwardNN(self.observation_space, 1, output_gain=1.0)

        self.log_std = nn.Parameter(
            torch.full((self.action_space,), self.initial_log_std, dtype=torch.float32)
        )

        self.action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(env.action_space.high, dtype=torch.float32)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self.actor_optimizer = Adam(
            list(self.actor.parameters()) + [self.log_std],
            lr=self.actor_learning_rate,
        )
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_learning_rate)

        self.obs_rms = RunningMeanStd(shape=(self.observation_space,))
        self.obs_clip = 10.0

        self.training_history = {
            "timesteps": [],
            "avg_ep_len": [],
            "avg_ep_return": [],
            "avg_ep_progress": [],
        }

    def _init_hyperparameters(self):
        self.iterations_per_batch = 16384 * 2
        self.max_iterations_per_episode = 8192
        self.max_episodes_per_batch = 8
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

        self.initial_log_std = -0.5

    def learn(self, num_iterations, best_actor_path=None, best_critic_path=None):
        t_so_far = 0
        best_metric = -float("inf")
        track_best = best_actor_path is not None and best_critic_path is not None
        while t_so_far < num_iterations:
            (
                batch_observations,
                batch_pre_squash,
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
                    mb_pre_squash = batch_pre_squash[mb_idx]
                    mb_old_log_probabilities = batch_log_probabilities[mb_idx]
                    mb_returns = batch_returns[mb_idx]
                    mb_advantages = batch_advantages[mb_idx]

                    values, current_log_probabilities, entropy = self.evaluate(mb_observations, mb_pre_squash)

                    ratio = torch.exp(current_log_probabilities - mb_old_log_probabilities)
                    surrogate1 = ratio * mb_advantages
                    surrogate2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * mb_advantages

                    actor_loss = -torch.min(surrogate1, surrogate2).mean() - self.entropy_coefficient * entropy.mean()
                    critic_loss = torch.nn.MSELoss()(values, mb_returns)

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_([self.log_std], self.max_grad_norm)
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
                f"{np.mean(batch_episode_lengths):.3f}",
                "avg_ep_return:",
                f"{np.mean(batch_episode_returns):.3f}",
                "avg_ep_progress:",
                f"{np.mean(batch_episode_progresses):.3f}",
                "log_std:",
                [f"{x:.3f}" for x in self.log_std.detach().tolist()],
            )

            avg_progress = float(np.mean(batch_episode_progresses))
            self.training_history["timesteps"].append(int(t_so_far))
            self.training_history["avg_ep_len"].append(float(np.mean(batch_episode_lengths)))
            self.training_history["avg_ep_return"].append(float(np.mean(batch_episode_returns)))
            self.training_history["avg_ep_progress"].append(avg_progress)

            if track_best and avg_progress > best_metric:
                best_metric = avg_progress
                self.save(best_actor_path, best_critic_path)
                print(f"  new best avg_ep_progress={avg_progress:.3f}, saved to {best_actor_path}")

            # =======================================================
            # EARLY STOPPING TRIPWIRE: Check if the target was reached
            # =======================================================
            max_progress_in_batch = max(batch_episode_progresses)
            if max_progress_in_batch >= 0.005:
                print(f"\n[TARGET REACHED] Run successfully completed in {t_so_far} total steps!")
                # Write this specific golden number to a clean file for your t-test
                with open("ppo_final_results.txt", "a") as f:
                    f.write(f"{t_so_far}\n")
                break # This instantly kills the run and lets your Python script move to Run 2
            # =======================================================

    def plot_training_metrics(self, output_dir="checkpoints/training_plots", show=False):
        timesteps = self.training_history["timesteps"]
        if len(timesteps) == 0:
            print("No training metrics available to plot.")
            return

        os.makedirs(output_dir, exist_ok=True)

        metric_specs = [
            ("avg_ep_len", "Average Episode Length"),
            ("avg_ep_return", "Average Episode Return"),
            ("avg_ep_progress", "Average Episode Progress"),
        ]

        for metric_key, title in metric_specs:
            plt.figure(figsize=(8, 4.5))
            plt.plot(timesteps, self.training_history[metric_key], linewidth=2)
            plt.title(f"{title} vs Total Timesteps")
            plt.xlabel("Total Timesteps")
            plt.ylabel(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            output_path = os.path.join(output_dir, f"{metric_key}.png")
            plt.savefig(output_path)

            if show:
                plt.show()
            plt.close()

    def rollout(self):
        batch_observations = []
        batch_pre_squash = []
        batch_log_probabilities = []
        batch_returns = []
        batch_advantages = []
        batch_episode_lengths = []
        batch_episode_returns = []
        batch_episode_progresses = []

        t_so_far = 0
        episodes = 0

        while t_so_far < self.iterations_per_batch and episodes < self.max_episodes_per_batch:
            episode_rewards = []
            episode_values = []
            episode_dones = []

            episode_return = 0.0
            observation, _ = self.env.reset()
            terminated = False
            truncated = False
            info = {}
            episode_i = -1

            for episode_i in range(self.max_iterations_per_episode):
                t_so_far += 1
                batch_observations.append(observation)

                observation_tensor = torch.tensor(observation, dtype=torch.float32)
                normalized_observation = self._normalize_observation(observation_tensor)
                value = self.critic(normalized_observation).item()

                action, pre_squash, log_probability = self.get_action(observation)
                observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_return += reward

                episode_rewards.append(reward)
                episode_values.append(value)
                episode_dones.append(done)

                batch_pre_squash.append(pre_squash)
                batch_log_probabilities.append(log_probability)

                if done or t_so_far >= self.iterations_per_batch:
                    break

            # Bootstrap if the episode wasn't terminated by the env (truncated, batch-cut, or step-cap)
            if terminated:
                last_value = 0.0
            else:
                with torch.no_grad():
                    last_obs_tensor = torch.tensor(observation, dtype=torch.float32)
                    last_value = self.critic(self._normalize_observation(last_obs_tensor)).item()

            episodes += 1
            episode_advantages, episode_returns = self.compute_gae(
                episode_rewards, episode_values, episode_dones, last_value=last_value
            )
            batch_advantages.extend(episode_advantages)
            batch_returns.extend(episode_returns)
            batch_episode_lengths.append(episode_i + 1)
            batch_episode_returns.append(episode_return)
            batch_episode_progresses.append(float(info.get("progress", 0.0)))

        batch_observations_np = np.array(batch_observations)
        self.obs_rms.update(batch_observations_np)

        batch_observations = torch.tensor(batch_observations_np, dtype=torch.float32)
        batch_pre_squash = torch.tensor(np.array(batch_pre_squash), dtype=torch.float32)
        batch_log_probabilities = torch.tensor(np.array(batch_log_probabilities), dtype=torch.float32)
        batch_returns = torch.tensor(np.array(batch_returns), dtype=torch.float32)
        batch_advantages = torch.tensor(np.array(batch_advantages), dtype=torch.float32)

        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-10)

        return (
            batch_observations,
            batch_pre_squash,
            batch_log_probabilities,
            batch_returns,
            batch_advantages,
            batch_episode_lengths,
            batch_episode_returns,
            batch_episode_progresses,
        )

    def _normalize_observation(self, observation_tensor):
        mean = torch.tensor(self.obs_rms.mean, dtype=torch.float32)
        std = torch.tensor(np.sqrt(self.obs_rms.var) + 1e-8, dtype=torch.float32)
        return torch.clamp((observation_tensor - mean) / std, -self.obs_clip, self.obs_clip)

    def _distribution(self, observations):
        normalized_observations = self._normalize_observation(observations)
        raw_mean = self.actor(normalized_observations)
        std = torch.exp(self.log_std).expand_as(raw_mean)
        return Independent(Normal(raw_mean, std), 1)

    def _squash_to_action(self, pre_squash):
        return self.action_bias + self.action_scale * torch.tanh(pre_squash)

    def get_action(self, observation):
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        distribution = self._distribution(observation_tensor)

        with torch.no_grad():
            pre_squash = distribution.sample()
            log_probability = distribution.log_prob(pre_squash)
            action = self._squash_to_action(pre_squash)

        return action.numpy(), pre_squash.numpy(), float(log_probability.item())

    def predict(self, observation, deterministic=True):
        observation_tensor = torch.tensor(observation, dtype=torch.float32)

        with torch.no_grad():
            normalized = self._normalize_observation(observation_tensor)
            raw_mean = self.actor(normalized)
            if deterministic:
                pre_squash = raw_mean
            else:
                std = torch.exp(self.log_std).expand_as(raw_mean)
                pre_squash = Normal(raw_mean, std).sample()
            action = self._squash_to_action(pre_squash)

        return action.numpy()

    def compute_gae(self, rewards, values, dones, last_value=0.0):
        advantages = []
        returns = []
        gae = 0.0
        next_value = last_value

        for i in reversed(range(len(rewards))):
            not_done = 1.0 - float(dones[i])
            delta = rewards[i] + self.gamma * next_value * not_done - values[i]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            next_value = values[i]

        return advantages, returns

    def evaluate(self, batch_observations, batch_pre_squash):
        distribution = self._distribution(batch_observations)
        log_probabilities = distribution.log_prob(batch_pre_squash)
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
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "log_std": self.log_std.detach(),
                "obs_rms_mean": self.obs_rms.mean,
                "obs_rms_var": self.obs_rms.var,
                "obs_rms_count": self.obs_rms.count,
            },
            actor_path,
        )
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, actor_path, critic_path=None):
        actor_blob = torch.load(actor_path, map_location="cpu", weights_only=False)
        if isinstance(actor_blob, dict) and "actor" in actor_blob:
            self.actor.load_state_dict(actor_blob["actor"])
            with torch.no_grad():
                self.log_std.copy_(actor_blob["log_std"])
            if "obs_rms_mean" in actor_blob:
                self.obs_rms.mean = actor_blob["obs_rms_mean"]
                self.obs_rms.var = actor_blob["obs_rms_var"]
                self.obs_rms.count = actor_blob["obs_rms_count"]
        else:
            self.actor.load_state_dict(actor_blob)
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
