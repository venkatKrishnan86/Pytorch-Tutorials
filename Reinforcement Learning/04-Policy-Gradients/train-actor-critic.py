# Still needs correction

import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import random

from utils import *

env = gym.make("CartPole-v1")

learning_rate = 0.001
gamma = 0.992
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.997
batch_size = 32
episodes = 1500

state_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.n

# policy function predictor
policy_network = PolicyNetwork(state_space_size, action_space_size)

# value function predictor
value_network = ValueNetwork(state_space_size)
value_criterion = nn.MSELoss()

policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=learning_rate)
value_optimizer = torch.optim.Adam(value_network.parameters(), lr=learning_rate)

for episode in range(episodes):
    state, _ = env.reset()
    state = torch.FloatTensor(state)
    done = False
    log_probs = []
    values = []
    next_values = []
    rewards = []

    while not done:
        # Exploration vs Exploitation
        if random.random() < epsilon:
            action = env.action_space.sample()
            log_prob = torch.log(1.0/torch.tensor(action_space_size))
        else:
            action_probs = torch.softmax(policy_network(state), 0)
            dist = Categorical(action_probs)
            action = dist.sample() # Instead of argmax
            log_prob = dist.log_prob(action)
            action = action.item() # To obtain the int value
        next_state, reward, done, _, _ = env.step(action)
        value = value_network(state)

        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)

        next_state = torch.FloatTensor(next_state)
        next_value = value_network(next_state)
        next_values.append(next_value)

        state = next_state
    
    returns_to_go = torch.FloatTensor(compute_rewards_to_go(rewards, gamma=gamma)).detach()
    values = torch.stack(values).squeeze()
    next_values = torch.stack(next_values).squeeze()
    log_probs = torch.stack(log_probs).squeeze()

    returns_to_go = returns_to_go + gamma*next_values

    advantage = returns_to_go - values

    policy_loss = -(log_probs * advantage).mean()
    value_loss: torch.Tensor = value_criterion(values, returns_to_go)

    policy_optimizer.zero_grad()
    policy_loss.backward(retain_graph=True)
    policy_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    print(f'Episode {episode}, Total Reward: {sum(rewards)}')
    
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

torch.save(policy_network.state_dict(), "cartpole_actor-critic_policy_net.pth")
torch.save(value_network.state_dict(), "cartpole_actor-critic_value_net.pth")
