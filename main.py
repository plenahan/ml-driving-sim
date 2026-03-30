import gymnasium as gym
import argparse
from ppo import PPO
from simulation.sim_env import SimEnv

def train(total_steps, actor_path, critic_path, plot_dir=None, show_plots=False):
    env = SimEnv(human=False)
    assert(type(env.observation_space) == gym.spaces.Box)
    assert(type(env.action_space) == gym.spaces.Box)
    model = PPO(env)
    model.learn(total_steps)
    if plot_dir is not None:
        model.plot_training_metrics(output_dir=plot_dir, show=show_plots)
    model.save(actor_path, critic_path)
    env.close()


def play(actor_path, critic_path=None, max_steps=20000):
    env = SimEnv(human=True)
    model = PPO(env)
    model.load(actor_path, critic_path)

    observation, _ = env.reset()
    done = False
    step_count = 0
    info = {}

    while not done and step_count < max_steps:
        action = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    print("episode_done", {"steps": step_count, "info": info})
    env.close()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--steps", type=int, default=20000)
    train_parser.add_argument("--actor-path", type=str, default="checkpoints/actor.pt")
    train_parser.add_argument("--critic-path", type=str, default="checkpoints/critic.pt")
    train_parser.add_argument("--plot-dir", type=str, default="checkpoints/training_plots")
    train_parser.add_argument("--show-plots", action="store_true")

    play_parser = subparsers.add_parser("play")
    play_parser.add_argument("--actor-path", type=str, default="checkpoints/actor.pt")
    play_parser.add_argument("--critic-path", type=str, default=None)
    play_parser.add_argument("--max-steps", type=int, default=20000)

    args = parser.parse_args()
    if args.mode == "train":
        train(args.steps, args.actor_path, args.critic_path, args.plot_dir, args.show_plots)
    elif args.mode == "play":
        play(args.actor_path, args.critic_path, args.max_steps)

if __name__ == "__main__":
    main()
