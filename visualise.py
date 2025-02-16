import numpy as np
from typing import Optional
from matplotlib import pyplot as plt


def smooth_array(r: np.ndarray, window_size: int = 10) -> np.ndarray:
    """Smooths an array using a moving average.

    Args:
        r (np.ndarray): The array to be smoothed.
        window_size (int, optional): The size of the moving average window. Defaults to 10.

    Returns:
        np.ndarray: The smoothed array.
    """
    kernel = np.ones(window_size) / window_size
    return np.convolve(r, kernel, mode="same")


def visualise_trian_results(
    dqn_reward: Optional[np.ndarray] = None, mcts_reward: Optional[np.ndarray] = None, smoothing_f: int = 15
) -> None:
    """Visualises the performance rewards using matplotlib.

    If both dqn_reward and mcts_reward are provided, both will be plotted.
    If only one is provided, only that reward curve is shown.

    Args:
        dqn_reward (Optional[np.ndarray]): The reward array for DQN. Defaults to None.
        mcts_reward (Optional[np.ndarray]): The reward array for MCTS. Defaults to None.
        smoothing_f (int, optional): The smoothing factor for the reward curves. Defaults to 15.
    """
    if dqn_reward is None and mcts_reward is None:
        print("No reward data provided. Exiting visualisation.")
        return

    plt.figure(figsize=(10, 6))

    if dqn_reward is not None:
        dqn_smoothed = smooth_array(dqn_reward, smoothing_f)
        plt.plot(dqn_smoothed, label="DQN", linewidth=2, linestyle="--", color="blue")

    if mcts_reward is not None:
        mcts_smoothed = smooth_array(mcts_reward, smoothing_f)
        plt.plot(mcts_smoothed, label="MCTS", linewidth=2, color="orange")

    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Cumulative Reward", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best", fontsize=12)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main function to load reward data and visualise the results."""
    try:
        dqn_reward = np.load("rewards_dqn.npy")
    except FileNotFoundError:
        dqn_reward = None
        print("DQN reward file not found.")

    try:
        mcts_reward = np.load("rewards_mcts.npy")
    except FileNotFoundError:
        mcts_reward = None
        print("MCTS reward file not found.")

    # Call visualise depending on available data
    visualise_trian_results(dqn_reward, mcts_reward, smoothing_f=15)


if __name__ == "__main__":
    main()
