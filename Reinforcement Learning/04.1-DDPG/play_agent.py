from matplotlib import animation
import matplotlib.pyplot as plt
import gym 
import torch

"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filelname>
"""

# Define the neural network model
from network import Actor, Critic
from utils import Agent

env = gym.make('LunarLanderContinuous-v2', render_mode='rgb_array')

# Hyperparameters
lr_actor = 2.5e-5
lr_critic = 2.5e-4
input_dims = env.observation_space.shape[0]
gamma = 0.99
tau = 0.001
warmup_percent = 0.015
buffer_size = 1e6
batch_size = 64
n_games = 1000
n_actions = env.action_space.shape[0]

agent = Agent(
    lr_actor,
    lr_critic,
    input_dims,
    gamma,
    tau,
    n_actions,
    warmup_percentage = warmup_percent,
    max_size = buffer_size,
    batch_size = batch_size
)

agent.actor.load_state_dict(torch.load('tmp/ddpg/actor_ddpg.pth'))
agent.critic.load_state_dict(torch.load('tmp/ddpg/critic_ddpg.pth'))
agent.actor.eval()
agent.critic.eval()

def save_frames_as_gif(frames, path='./', filename='tmp/output-no-noise.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

#Make gym env
# env = gym.make('CartPole-v1')

#Run the env
state = env.reset()[0]
frames = []
for t in range(1000):
    #Render to frames buffer
    state = torch.Tensor(state).to(agent.actor.device)
    frames.append(env.render())
    action = agent.choose_action(state, train=True)
    state, _, done, _, _ = env.step(action)
    if done:
        break
save_frames_as_gif(frames, filename="tmp/output.gif")

state = env.reset()[0]
frames = []
for t in range(1000):
    #Render to frames buffer
    state = torch.Tensor(state).to(agent.actor.device)
    frames.append(env.render())
    action = agent.choose_action(state, train=False)
    state, _, done, _, _ = env.step(action)
    if done:
        break

env.close()
save_frames_as_gif(frames, filename="tmp/output-no-noise.gif")