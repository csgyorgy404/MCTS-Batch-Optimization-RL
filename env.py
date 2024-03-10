import gymnasium as gym


def create_env(env_name, render_mode='human', seed=42):
    try:
        environment = gym.make(env_name, render_mode=render_mode)
    except gym.error.NameNotFound:
        raise ValueError(f"{env_name} environment is not implemented or"
                            f" not found in the official gym repository")
    else:
        _, __ = environment.reset(seed=seed)

    in_features = environment.observation_space.shape[0]
    out_features = environment.action_space.n

    return environment, in_features, out_features
