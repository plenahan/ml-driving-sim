import gymnasium as gym
from ppo import PPO

def main():
    env = gym.make('LunarLanderContinuous-v3', render_mode='human')
    assert(type(env.observation_space) == gym.spaces.Box)
    assert(type(env.action_space) == gym.spaces.Box)
    model = PPO(env)
    model.learn(20000000)
    print("hi")

if __name__ == "__main__":
    main()
