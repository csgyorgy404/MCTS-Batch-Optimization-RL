# MCTS-Based Batch Optimization for Reinforcement Learning

This repository contains the implementation and experimental results from the publication:

**"MCTS-Based Policy Improvement for Reinforcement Learning"**

## Overview
Traditional reinforcement learning (RL) batch sampling methods often use random or uniform strategies, which may not prioritize the most informative experiences. This research proposes an alternative approach that integrates **Monte Carlo Tree Search (MCTS)** into the batch selection process of RL algorithms. By leveraging MCTS, the method systematically selects more informative trajectories, leading to improved policy learning, faster convergence, and overall enhanced performance.

## Features
- **MCTS-Driven Batch Optimization**: Uses MCTS to prioritize the most valuable experience batches for training.
- **Improved Policy Learning**: Accelerates policy convergence by focusing on informative samples.
- **Experiments on RL Benchmarks**: Evaluated on standard OpenAI Gym environments including **CartPole, MountainCar, Acrobot, Taxi-v3, CliffWalking, and HighwayEnv**.
- **Comparison with Traditional Methods**: Demonstrates superior performance over uniform batch sampling.

## Installation
To set up the environment, clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/MCTS-Batch-Optimization-RL.git
cd MCTS-Batch-Optimization-RL
pip install -r requirements.txt
```

## Experimental Results
The proposed method has been extensively tested on OpenAI Gym environments. The results indicate that MCTS-guided batch selection improves training efficiency by:
- Reducing the number of gradient updates needed for convergence.
- Enhancing final policy performance across various environments.
- Outperforming traditional uniform batch sampling methods.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
