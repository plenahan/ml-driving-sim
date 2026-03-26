import gymnasium as gym
from ppo import PPO
from simulation.sim_env import SimEnv
import argparse

def test(max_steps, actor_path, critic_path):
    env = SimEnv(human=True)

    model = PPO(env)
    model.load(actor_path, critic_path)
    obs = env.reset()
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Initial observation:", obs)
    done = False
    step_count = 0
    info = {}

    while not done and step_count < max_steps:
        action = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    print("episode_done", {"steps": step_count, "info": info})
    env.close()

def train(steps, actor_path, critic_path):
    env = SimEnv(human=True)
    model = PPO(env)
    model.learn(steps)
    model.save(actor_path, critic_path)
    env.close()

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--steps", type=int, default=2000000)
    train_parser.add_argument("--actor-path", type=str, default="checkpoints/actor.pt")
    train_parser.add_argument("--critic-path", type=str, default="checkpoints/critic.pt")

    play_parser = subparsers.add_parser("test")
    play_parser.add_argument("--actor-path", type=str, default="checkpoints/actor.pt")
    play_parser.add_argument("--critic-path", type=str, default=None)
    play_parser.add_argument("--max-steps", type=int, default=20000)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.steps, args.actor_path, args.critic_path)
    elif args.mode == "test":
        test(args.max_steps, args.actor_path, args.critic_path)

if __name__ == "__main__":
    main()
