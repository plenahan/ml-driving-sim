import gymnasium as gym
from ppo import PPO
from simulation.sim_env import SimEnv

def main():
    env = SimEnv()
    #assert(type(env.observation_space) == gym.spaces.Box)
    #assert(type(env.action_space) == gym.spaces.Box)
    #model = PPO(env)
    #model.learn(20000000)
    #print("hi")

if __name__ == "__main__":
    main()
