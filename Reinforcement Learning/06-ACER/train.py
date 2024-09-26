# Special thanks to GPT, major part is coded by it

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define the neural network model
from utils import QNetwork

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Actions: 0, 1 (L, R)

# Define hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.997
batch_size = 32
memory = deque(maxlen=2000)
episodes = 2000

# Create the Q-network and target Q-network
input_size = env.observation_space.shape[0]
action_space_size = env.action_space.n
q_network = QNetwork(input_size, action_space_size)
target_q_network = QNetwork(input_size, action_space_size)
target_q_network.load_state_dict(q_network.state_dict())
target_q_network.eval()

# Define the loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

# Training the agent
for episode in range(episodes):
    state, _ = env.reset()
    state = torch.Tensor(state)
    total_reward = 0

    while True:
        # Exploration vs. exploitation
        if random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            q_values = q_network(state)
            action = torch.argmax(q_values).item()

        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.Tensor(next_state)
        total_reward += reward

        memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            break

    # Train the agent after each episode only if enough examples exists in the memory
    if len(memory) >= batch_size:
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                with torch.no_grad():
                    target = reward + gamma * torch.max(target_q_network(next_state))
            else:
                target = reward

            # For computing Q* at step t+1 (torch.no_grad not necessary, it works better without it)
            q_values = q_network(state)
            q_values[action] = target       # This will ensure in the MSE Loss, the other dimensions go to zero, only the action dimension is used for loss

            optimizer.zero_grad()
            loss = criterion(q_values, q_network(state))
            loss.backward()
            optimizer.step()

    # Update the target Q-network periodically
    if episode % 10 == 0:
        target_q_network.load_state_dict(q_network.state_dict())

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Save the trained model
torch.save(q_network.state_dict(), "cartpole_dqn.pth")
