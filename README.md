
# Unsupervised Learning and Reinforcement Learning Project

This project demonstrates the implementation of both Unsupervised Learning (K-Means clustering) and Reinforcement Learning (Deep Q-Network on CartPole environment).

## Unsupervised Learning
- The unsupervised learning folder contains code for customer segmentation using K-Means clustering.
- It uses customer data to segment users into different groups based on their spending habits.

## Reinforcement Learning
- The reinforcement learning folder contains a Deep Q-Learning agent that plays the CartPole game.
- The agent learns to balance a pole on a moving cart by maximizing the cumulative reward.

## Folder Structure:
- `data/`: Placeholder for datasets used in both unsupervised and reinforcement learning examples.
- `models/`: Pretrained models can be stored here.
- `notebooks/`: Contains Jupyter notebooks for the unsupervised and reinforcement learning tasks.

## How to Run
1. Install the necessary dependencies:
    ```
    pip install tensorflow numpy pandas matplotlib gym scikit-learn
    ```
2. For unsupervised learning, run the `unsupervised_learning/kmeans_clustering.py` script.
3. For reinforcement learning, run the `reinforcement_learning/cartpole_dqn.py` script.

## License
MIT License
