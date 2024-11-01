import torch
from torch import nn
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt 
import numpy as np
import os
import imageio

# Defining the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_space_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x  

# Defining the value network
class ValueNetwork(nn.Module):
    def __init__(self, state_space_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_space_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x   

def compute_rewards_to_go(rewards, gamma = 0.99):
    rewards_to_go = []
    last_reward = 0
    for r in rewards[::-1]:
        last_reward = r + gamma*last_reward
        rewards_to_go.insert(0, last_reward)
    return rewards_to_go

def save_random_agent_gif(env, model, num = 5):
    total_reward = 0 
    for i in range(num):
        # frames = []
        state, _ = env.reset()       
        while True:
            state = torch.Tensor(state)
            with torch.no_grad():
                q_values = model(state)
            action = torch.argmax(q_values).item()

            # frame = env.render(mode='rgb_array')
            # frames.append(_label_with_episode_number(frame, episode_num=i))

            state, reward, done, _,_ = env.step(action)
            total_reward+=reward
            if done:
                break
        # imageio.mimwrite(os.path.join('./videos/', 'cartpole_test_'+str(i+1)+'.gif'), frames, fps=60)
    env.close()
    # imageio.mimwrite(os.path.join('./videos/', 'cartpole_test.gif'), frames, fps=60)
    return total_reward/num
