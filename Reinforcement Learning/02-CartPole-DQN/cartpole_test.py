import gym
import torch

# Define the neural network model
from utils import QNetwork, save_random_agent_gif

env = gym.make('CartPole-v1', render_mode='human')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
q_network = QNetwork(input_size, output_size)

q_network.load_state_dict(torch.load('./cartpole_dqn.pth'))
q_network.eval()

avg_total_reward = save_random_agent_gif(env, q_network, 1)

print(f"Test Total Reward: {avg_total_reward}")