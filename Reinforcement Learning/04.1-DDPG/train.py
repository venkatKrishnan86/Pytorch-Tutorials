# We shall implement the training loop here
import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
import wandb

from utils import Agent

env = gym.make('LunarLanderContinuous-v2')

# Hyperparameters
lr_actor = 2.5e-5
lr_critic = 2.5e-4
input_dims = env.observation_space.shape[0]

# The low and high value is -1 and 1
n_actions = env.action_space.shape[0]

print("State dims:", input_dims)
print("Action dims:", n_actions)

gamma = 0.99
tau = 0.001
warmup_percent = 0.015
buffer_size = 1e6
batch_size = 64
n_games = 1000

# Logging into wandb
wandb.login()

# Initialize Weights and Biases
run = wandb.init(
    project="ddpg-lunar-lander",
    config={
        "lr_actor": lr_actor,
        "lr_critic": lr_critic,
        "batch_size": batch_size,
        "num_episodes": n_games,
        "gamma": gamma,
        "tau": tau,
        "warmup_percent": warmup_percent,
        "buffer_size": buffer_size,
        "input_dims": input_dims,
    })

# DDPG Algorithm
# --------------
# Randomly initialize critic network Q(s, a|θQ) and actor μ(s|θμ) with weights θQ and θμ. 
# Initialize target network Q′ and μ′ with weights θQ′ ← θQ, θμ′ ← θμ 
# Initialize replay buffer R
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

seed = 0

print("Training: DDPG")

score_history = []

for i in range(n_games):
    # Initialize the noise process for action exploration
    agent.noise.reset()

    # Receive initial observation state s1
    state = env.reset()[0]
    state = torch.tensor(state, dtype=torch.float32).to(agent.actor.device)

    done = False
    score = 0
    while not done:
        # Select action at = μ(st|θμ) + Nt according to the current policy and exploration noise
        action = agent.choose_action(state)

        # Execute action at and observe reward rt and observe new state st+1
        new_state, reward, done, info, _ = env.step(action)

        # Store transition (st, at, rt, st+1) in R
        agent.remember(state, action, reward, new_state, int(done))

        # Sample a random minibatch of N transitions (si, ai, ri, si+1) from R
        # Calculate y = ri + γQ'(si+1, μ'(si+1|θμ')|θQ')
        # Update critic by minimizing the loss L = 1/N * Σ(yi - Q(si, ai|θQ))^2
        # Update the actor policy using the sampled policy gradient
        # Update the target networks: θQ' = τθQ + (1 - τ)θQ', θμ' = τθμ + (1 - τ)θμ'
        agent.learn(run)

        score += reward
        state = torch.tensor(new_state, dtype=torch.float32).to(agent.actor.device)
    score_history.append(score)
    run.log({
        "episode": i+1,
        "score": score, 
        "100_game_avg": np.mean(score_history[-100:])
    })
    print("Episode: ", i, "Score: %.1f" % score)
    print("100 game average: %.1f" % np.mean(score_history[-100:]))

    if i % 25 == 0:
        agent.save_models()

filename = 'LunarLanderContinuous.png'
plt.plot(score_history)
plt.title('LunarLanderContinuous-v2')
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.savefig(filename)
# plotLearning(score_history, filename, window=100)
env.close()
