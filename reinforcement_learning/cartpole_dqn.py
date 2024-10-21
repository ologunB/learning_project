
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Hyperparameters
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32

# Q-Network
model = tf.keras.Sequential([
    layers.Dense(24, activation='relu', input_shape=(env.observation_space.shape[0],)),
    layers.Dense(24, activation='relu'),
    layers.Dense(env.action_space.n, activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

# Experience replay memory
memory = deque(maxlen=2000)

def choose_action(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    q_values = model.predict(state)
    return np.argmax(q_values[0])

def train_model(batch_size):
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in batch:
        target = reward
        if not done:
            target += gamma * np.amax(model.predict(next_state)[0])
        target_q_values = model.predict(state)
        target_q_values[0][action] = target
        model.fit(state, target_q_values, epochs=1, verbose=0)

# Training loop
for episode in range(1000):
    state = env.reset().reshape(1, env.observation_space.shape[0])
    for time_step in range(500):
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state.reshape(1, env.observation_space.shape[0])
        reward = reward if not done else -10
        memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print(f"Episode: {episode+1}, Score: {time_step}")
            break
    train_model(batch_size)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
