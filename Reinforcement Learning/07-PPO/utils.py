import torch
from torch import nn
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt 
import numpy as np
import os
import imageio

# Define the neural network model
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x   


def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)

    return im


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
