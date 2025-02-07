import gymnasium
import numpy as np

env = None

def init_env(id):
    global env
    env = gymnasium.make("Pendulum-v1")
    env.reset(seed=0)

def reset_env():
    obs, _ = env.reset()
    return obs.tolist()

def step_env(action):
    obs, reward, terminated, truncated, info = env.step(action)
    return obs.tolist(), float(reward), bool(terminated), bool(truncated)


if __name__ == "__main__":
    init_env(0)

    obs = reset_env()
    print(obs)

    # for i in range(100):
    #     action = [1]
    #     obs, reward, terminated, truncated = step_env(action)
    #     print(obs)

