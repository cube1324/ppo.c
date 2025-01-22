import gymnasium
import numpy as np

global env

def init_env(id):
    global env
    env = gymnasium.make("Pendulum-v1")

def step_env(action):
    global env
    
    obs, reward, terminated, truncated, _ = env.step(action)
    return obs.tolist(), float(reward), terminated, truncated

def reset_env():
    global env
    obs, _ =  env.reset()
    return obs.tolist()



if __name__ == "__main__":
    init_env("CartPole-v1")
    print(reset_env())
    print(step_env([1.0]))