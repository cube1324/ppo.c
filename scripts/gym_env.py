import gymnasium
import numpy as np

env = None

def init_env(env_id, seed):
    global env

    print("Initializing environment...")
    print("env_id:", env_id)
    if env_id == 0:
        env = gymnasium.make("Pendulum-v1")
        env.reset(seed=seed)

    elif env_id == 1:
        env = gymnasium.make("BipedalWalker-v3")
        env.reset(seed=seed)

    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    horizon = env.spec.max_episode_steps

    print("action_size:", action_size)
    print("state_size:", state_size)
    print("horizon:", horizon)
    return action_size, state_size, horizon

def reset_env():
    obs, _ = env.reset()
    return obs.tolist()

def step_env(action):
    # print("action:", action)
    obs, reward, terminated, truncated, info = env.step(action)
    # print("obs:", obs)
    # print("reward:", reward)
    # print("terminated:", terminated)
    # print("truncated:", truncated)
    # print("--"*20)
    return obs.tolist(), float(reward), bool(terminated), bool(truncated)


if __name__ == "__main__":
    init_env(1, 0)

    obs = reset_env()
    print(obs)

    # for i in range(100):
    #     action = [1]
    #     obs, reward, terminated, truncated = step_env(action)
    #     print(obs)

