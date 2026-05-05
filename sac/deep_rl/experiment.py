"""
Experiment class for running experiments

- rollouts
- graph drawing

"""
import json
import os
from datetime import datetime

import numpy as np
import torch as th
from cycler import cycler
from gymnasium import Env

from deep_rl import timed_decorator
from learner import SoftActorCriticLearner
from replay_buffer import ReplayBuffer
from runner import Runner
from controller import Controller
import matplotlib.pyplot as plt


class MultiExperiment:

    def __init__(self, n, env, experiment_class, model_class, model_constructor, config_params):
        self.config_params = config_params
        number_of_graph_points = int(self.config_params['max_steps'] / self.config_params['runner_steps'])
        self.n = n

        self.experiments_buffer_shapes = {'n_rewards': (number_of_graph_points,),
                                          'n_episode_lengths': (number_of_graph_points,),
                                          'n_episode_infos': (number_of_graph_points,),
                                          'n_loss_means': (number_of_graph_points, 4),
                                          'n_loss_stds': (number_of_graph_points, 4),
                                          'n_env_steps': (number_of_graph_points,),
                                          'n_episode_durations': (number_of_graph_points,)}
        self.experiments_buffer = ReplayBuffer(n, self.experiments_buffer_shapes)
        self.env = [env for _ in range(n)]
        self.models = [model_class(*model_constructor) for _ in range(n)]
        self.experiment_class = experiment_class
        self.experiment_folder_path = self.config_params['save_path'] + datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.experiment_folder_path)
        json.dump(self.config_params, open(f'{self.experiment_folder_path}/config.json', 'w'))

    def run_experiments(self):
        for i in range(self.n):
            self.config_params['best_model_path'] = f'{self.experiment_folder_path}/best_actor_{i}.pth'
            experiment = self.experiment_class(self.env[i], model=self.models[i], config_params=self.config_params,
                                               render_graphs=self.config_params.get('render_graphs', True))
            try:
                fig, rewards, episode_lengths, episode_infos, episode_loss_means, episode_loss_stds, env_steps, episode_durations = experiment.run()
                self.experiments_buffer.store(n_rewards=rewards, n_episode_lengths=episode_lengths, n_episode_infos=episode_infos,
                                              n_loss_means=episode_loss_means, n_loss_stds=episode_loss_stds,
                                              n_env_steps=env_steps,
                                              n_episode_durations=np.array([sum(episode_durations[:i]) for i in range(len(episode_durations))]))
                th.save(experiment.learner.model, f'{self.experiment_folder_path}/experiment_{i}.pth')
                self.experiments_buffer.save_to_file(f'{self.experiment_folder_path}/results_{i}_{np.round(np.max(rewards[-2000:]), 2)}_{np.round(np.max(episode_infos[-2000:]), 2)}.json')
                fig.savefig(f'{self.experiment_folder_path}/experiment_{i}.png')
                plt.close(fig)
            except KeyboardInterrupt:
                experiment.close()
        plt.ioff()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        n_env_steps = self.experiments_buffer['n_env_steps']
        n_rewards = self.experiments_buffer['n_rewards']
        n_loss_means = self.experiments_buffer['n_loss_means']
        n_episode_durations = self.experiments_buffer['n_episode_durations']
        n_episode_infos = self.experiments_buffer['n_episode_infos']

        for i in range(self.n):
            axes[0].plot(n_env_steps[i], n_rewards[i], c='b', alpha=0.1)
        axes[0].fill_between(np.mean(n_env_steps, axis=0), np.mean(n_rewards, axis=0) - np.std(n_rewards, axis=0),
                             np.mean(n_rewards, axis=0) + np.std(n_rewards, axis=0), alpha=0.4)
        axes[0].plot(np.mean(n_env_steps, axis=0), np.mean(n_rewards, axis=0), c='r')
        axes[0].set_title('Mean Total Rewards')

        axes[1].set_prop_cycle(cycler('color', ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']))
        for i in range(self.n):
            axes[1].plot(n_env_steps[i], n_loss_means[i], alpha=0.1)
        axes[1].plot(np.mean(n_env_steps, axis=0), np.mean(n_loss_means, axis=0))
        axes[1].set_title('Mean Losses')

        for i in range(self.n):
            axes[2].plot(n_env_steps[i], n_episode_infos[i], c='b', alpha=0.1)
        axes[2].plot(np.mean(n_env_steps, axis=0), np.mean(n_episode_infos, axis=0), c='tab:blue')
        axes[2].set_title('Mean Episode Infos')
        plt.savefig(f'{self.experiment_folder_path}/all_results.png')
        # plt.show()
        plt.close(fig)
        return self.experiment_folder_path


class Experiment:

    def __init__(self, config_params, render_graphs=True):
        self.learner = None
        self.controller = None
        self.runner = None

        self.config_params = config_params
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_infos = {}
        self.episode_loss_means = []
        self.episode_loss_stds = []
        self.env_steps = []
        self.episode_runner_durations = []
        self.episode_learn_durations = []
        self.episode_durations = []
        self.temp_time = datetime.now()
        self.start_time = datetime.now()

        self.render_graphs = render_graphs
        if self.render_graphs:
            plt.ion()
            self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))

            self.axs[0].plot(self.env_steps, self.episode_rewards)
            self.axs[0].set_prop_cycle(cycler('color', ['tab:blue']))
            self.axs[0].set_ylabel('Episode Sum of Rewards')
            self.info_ax, self.info_labels = None, None

            self.axs[1].plot(self.env_steps, self.episode_lengths)
            self.axs[1].set_prop_cycle(cycler('color', ['tab:blue']))
            self.axs[1].set_ylabel('Episode Lengths')

            labels = ['Policy', 'Critic_1', 'Critic_2', 'Alpha']
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

            # Create and plot the lines using a list comprehension and zip the labels and colors
            for label, color in zip(labels, colors):
                self.axs[2].plot([0], [0], label=label, color=color)
            self.axs[2].set_ylabel('Episode Losses')
            self.axs[2].legend()

            for ax in self.axs:
                ax.set_xlabel('Environment Steps')
            self.first_run = True
            plt.tight_layout()
            plt.pause(1)

    def plot(self, step):
        if not self.render_graphs:
            return

        if self.first_run and len(self.episode_infos) > 0:
            self.info_ax = self.axs[0].twinx()
            for key in self.episode_infos.keys():
                self.info_ax.plot(self.env_steps, self.episode_infos[key], label=key, alpha=0.5, color='tab:orange')
            self.info_ax.legend()
            plt.tight_layout()

        if self.config_params['render'] and len(self.episode_rewards) > 0:
            if step % self.config_params['render_every'] == 0:
                self.axs[0].lines[0].set_xdata(self.env_steps)
                self.axs[0].lines[0].set_ydata(self.episode_rewards)
                self.axs[0].draw_artist(self.axs[0].lines[0])

                self.axs[1].lines[0].set_xdata(self.env_steps)
                self.axs[1].lines[0].set_ydata(self.episode_lengths)
                self.axs[1].draw_artist(self.axs[1].lines[0])

                for i in range(len(self.axs[2].lines)):
                    self.axs[2].lines[i].set_xdata(self.env_steps)
                    self.axs[2].lines[i].set_ydata(np.asarray(self.episode_loss_means)[:, i])
                    self.axs[2].draw_artist(self.axs[2].lines[i])

                if self.info_ax is not None:
                    for line, key in zip(self.info_ax.lines, self.episode_infos.keys()):
                        line.set_xdata(self.env_steps)
                        line.set_ydata(self.episode_infos[key])
                    for line in self.info_ax.lines:
                        self.info_ax.draw_artist(line)
                    self.info_ax.relim(visible_only=True)
                    self.info_ax.autoscale_view(scalex=True, scaley=True)

                # Update the axes limits
                for ax in self.axs:
                    ax.relim(visible_only=True)
                    ax.autoscale_view(scalex=True, scaley=True)

                self.fig.canvas.blit(self.fig.bbox)
                self.fig.canvas.flush_events()

    def record(self, returns, rollout_lengths, rollout_infos, episode_loss_mean, episode_loss_std, step, run_duration, learn_duration):
        if len(rollout_infos) > 0:
            for key in rollout_infos.shape.keys():
                key_values_mean = np.asarray(rollout_infos[key]).mean()
                if key not in self.episode_infos.keys():
                    self.episode_infos[key] = []
                self.episode_infos[key].append(key_values_mean)

        self.episode_loss_means.append(episode_loss_mean)
        self.episode_loss_stds.append(episode_loss_std)
        self.env_steps.append(step)
        self.episode_runner_durations.append(run_duration)
        self.episode_learn_durations.append(learn_duration)
        self.episode_durations.append(run_duration + learn_duration)
        # Calculate the mean of episode rewards and lengths if provided
        mean_rewards = np.mean([np.sum(ep) for ep in returns]) if returns is not None else None
        mean_lengths = np.asarray(rollout_lengths).mean() if rollout_lengths is not None else None
        if len(self.episode_rewards) == 0 and mean_rewards is not None and mean_lengths is not None:
            num_entries_to_add = len(self.episode_loss_means) - 1
            self.episode_rewards.extend([mean_rewards] * num_entries_to_add)
            self.episode_lengths.extend([mean_lengths] * num_entries_to_add)
        if len(self.episode_rewards) > 0:
            if mean_rewards is None and mean_lengths is None:
                self.episode_rewards.append(self.episode_rewards[-1])
                self.episode_lengths.append(self.episode_lengths[-1])
            else:
                self.episode_rewards.append(mean_rewards)
                self.episode_lengths.append(mean_lengths)

        if step % self.config_params['print_every'] == 0:
            avg_ep_len = self.episode_lengths[-1] if self.episode_lengths else float('nan')
            avg_ep_return = self.episode_rewards[-1] if self.episode_rewards else float('nan')
            avg_ep_progress = np.asarray(rollout_infos['progress']).mean() if (rollout_infos is not None and len(rollout_infos) > 0 and 'progress' in rollout_infos.shape) else float('nan')
            losses = self.episode_loss_means[-1]
            print(
                f"t_so_far: {step}",
                f"avg_ep_len: {avg_ep_len:.3f}",
                f"avg_ep_return: {avg_ep_return:.3f}",
                f"avg_ep_progress: {avg_ep_progress:.3f}",
                f"actor_loss: {losses[0]:.4f}",
                f"q_loss: {(losses[1] + losses[2]) / 2:.4f}",
                f"alpha_loss: {losses[3]:.4f}",
            )

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)

    def run(self):
        if len(self.episode_infos) == 0:
            self.episode_infos = {'info': [np.linspace(0, len(self.episode_rewards), len(self.episode_rewards))]}
        return self.fig, self.episode_rewards, self.episode_lengths, self.episode_infos[list(self.episode_infos.keys())[0]], self.episode_loss_means, self.episode_loss_stds, self.env_steps, self.episode_durations


class SACExperiment(Experiment):

    def __init__(self, env: Env, model: th.nn.Module, config_params, render_graphs=True):
        super().__init__(config_params, render_graphs=render_graphs)
        model.to(config_params['device'])
        self.env = env
        self.learner = SoftActorCriticLearner(model, self.env.action_space.shape, config_params)
        self.controller = Controller(model, config_params)
        self.runner = Runner(env, self.controller, config_params)
        self.replay_buffer_shapes = {'states': self.env.observation_space.shape,
                                     'actions': self.env.action_space.shape,
                                     'rewards': (1,),
                                     'next_states': self.env.observation_space.shape,
                                     'dones': (1,)}
        self.replay_buffer = ReplayBuffer(config_params['replay_buffer_size'], self.replay_buffer_shapes, config_params['batch_size'], config_params['device'])
        self.grad_repeats = config_params['grad_repeats']

    @timed_decorator
    def _learn_from_episodes(self, episodes_buffer):
        self.replay_buffer.append(episodes_buffer)
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
        
        loss = []
        for _ in range(self.grad_repeats):
            batch_buffer = self.replay_buffer.sample()
            policy_loss, q1_loss, q2_loss, alpha_loss = self.learner.train(batch_buffer)
            loss.append([policy_loss, q1_loss, q2_loss, alpha_loss])
        return np.mean(loss, axis=0), np.std(loss, axis=0)

    def run(self):
        episode_buffer, _, _, _, runner_duration = self.runner.run_steps(step:=self.config_params['warmup_steps'])
        self.replay_buffer.append(episode_buffer)
        print(f"Warmed up replay buffer with {len(self.replay_buffer)} samples and took {runner_duration} seconds.")

        best_progress = -float('inf')
        best_model_path = self.config_params.get('best_model_path')

        while step < self.config_params['max_steps']:
            episode_buffer, rollout_returns, rollout_lengths, rollout_infos, runner_duration = self.runner.run_steps(self.config_params['runner_steps'])
            episode_loss_mean, episode_loss_std, learn_duration = self._learn_from_episodes(episode_buffer)
            step += self.config_params['runner_steps']
            self.record(rollout_returns, rollout_lengths, rollout_infos, episode_loss_mean, episode_loss_std, step, runner_duration, learn_duration)
            self.plot(step)
            self.first_run = False

            if best_model_path is not None and rollout_infos is not None and len(rollout_infos) > 0 and 'progress' in rollout_infos.shape:
                avg_ep_progress = float(np.asarray(rollout_infos['progress']).mean())
                if avg_ep_progress > best_progress:
                    best_progress = avg_ep_progress
                    th.save(self.learner.model.actor.state_dict(), best_model_path)
                    print(f"  new best avg_ep_progress={avg_ep_progress:.3f}, saved to {best_model_path}")

        return super().run()
