# We shall implement the class for exploration noise and the replay buffer here
import torch
from torch import nn
import random
from network import Actor, Critic

class OUActionNoise:
    """
        Ornstein-Uhlenbeck process for the exploration noise
    """
    def __init__(
        self, 
        mu, 
        theta = 0.15, 
        sigma = 0.2, 
        dt = 1e-2, 
        x0 = None
    ):
        """
            Arguments:
            ----------
            mu: torch.Tensor
                The mean value of the noise

            theta: float
                The rate of mean reversion

            sigma: float
                The volatility parameter

            dt: float
                The time step

            x0: torch.Tensor
                The initial value of the noise
        """
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0

        # Initialize the state
        # To reset the temporal correlation from the previous episode 
        self.reset()

    def reset(self):
        """
            Reset the state of the noise
        """
        self.x_prev = self.x0 if self.x0 is not None else torch.zeros_like(self.mu)
    
    def __call__(self):
        """
            Description:
            ------------
            Generate the OU-noise (which is correlated in time)

            Returns:
            --------
            x: torch.Tensor
                The generated noise
        """ 
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * (self.dt ** 0.5) * torch.randn_like(self.mu)
        self.x_prev = x
        return x

class ReplayBuffer:
    """
        Replay buffer for storing the experiences
    """
    def __init__(self, capacity = 1e6, warmup_percentage = 0.75):
        """
            Arguments:
            ----------
            capacity: int
                The maximum capacity of the replay buffer
            
            warmup_percentage: float
                The percentage of the capacity to fill before starting to sample mini-batches
        """
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_state_buffer = []
        self.terminal_buffer = []
        self.capacity = capacity
        self.warmup = int(warmup_percentage * capacity)

    def add(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        reward: torch.Tensor, 
        next_state: torch.Tensor, 
        done: torch.Tensor
    ):
        """
            Add the experience to the replay buffer
        """
        if len(self.state_buffer) == self.capacity:
            self.state_buffer.pop(0)
            self.action_buffer.pop(0)
            self.reward_buffer.pop(0)
            self.next_state_buffer.pop(0)
            self.terminal_buffer.pop(0)
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.next_state_buffer.append(next_state)
        self.terminal_buffer.append(1 - done)

    def sample(self, batch_size):
        """
            Sample a mini-batch of experiences from the replay buffer

            Returns:
            --------
            mini_batch: tuple
                The tuple of mini-batch experiences
        """
        if len(self.state_buffer) < self.warmup:
            return None
        state_tensor = torch.stack(self.state_buffer)
        action_tensor = torch.stack(self.action_buffer)
        reward_tensor = torch.stack(self.reward_buffer)
        next_state_tensor = torch.stack(self.next_state_buffer)
        terminal_tensor = torch.stack(self.terminal_buffer)

        num_range = list(range(len(self.state_buffer)))
        batch = torch.tensor(random.sample(num_range, batch_size), dtype = torch.long)

        state_tensor = state_tensor[batch]
        action_tensor = action_tensor[batch]
        reward_tensor = reward_tensor[batch]
        next_state_tensor = next_state_tensor[batch]
        terminal_tensor = terminal_tensor[batch]

        return state_tensor, action_tensor, reward_tensor, next_state_tensor, terminal_tensor
    
    def __len__(self):
        return len(self.state_buffer)
    

class Agent:
    def __init__(
        self,
        lr_actor,
        lr_critic,
        state_dims,
        gamma = 0.99,
        tau = 0.001,
        n_actions = 2,
        max_size = 1e6,
        warmup_percentage = 0.1,
        fc1_dims = 400,
        fc2_dims = 300,
        batch_size = 64
    ):
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size, warmup_percentage)
        self.batch_size = batch_size
        self.tau = tau

        self.actor = Actor(
            lr_actor, 
            state_dims, 
            fc1_dims, 
            fc2_dims, 
            n_actions,
            name='actor'
        )
        self.critic = Critic(
            lr_critic, 
            state_dims, 
            fc1_dims, 
            fc2_dims, 
            n_actions,
            name='critic'
        )
        self.target_actor = Actor(
            lr_actor, 
            state_dims, 
            fc1_dims, 
            fc2_dims, 
            n_actions,
            name='target_actor'
        )
        self.target_critic = Critic(
            lr_critic, 
            state_dims, 
            fc1_dims, 
            fc2_dims, 
            n_actions,
            name='target_critic'
        )
        self.noise = OUActionNoise(mu = torch.zeros(n_actions))

        # To copy the exact parameters of the actor and critic networks to the target networks
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau = None):
        """
            Update the target network parameters
            theta_target <-- tau * theta_local + (1 - tau) * theta_target

            where tau << 1
        """
        if tau is None:
            tau = self.tau

        # Update the target actor network
        for target_actor_param, actor_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_actor_param.data.copy_(tau * actor_param.data + (1 - tau) * target_actor_param.data)
        
        # Update the target critic network
        for target_critic_param, critic_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_critic_param.data.copy_(tau * critic_param.data + (1 - tau) * target_critic_param.data)

    def choose_action(
        self, 
        state: torch.Tensor,
        train = True
    ):
        self.actor.eval()
        state = state.to(self.actor.device)
        if train:
            action = self.actor(state) + self.noise().to(self.actor.device)
        else:
            action = self.actor(state)
        self.actor.train()

        return action.cpu().detach().numpy()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(
            state, 
            torch.tensor(action, dtype = torch.float32), 
            torch.tensor(reward, dtype = torch.float32), 
            torch.tensor(next_state, dtype = torch.float32), 
            torch.tensor(done, dtype = torch.float32)
        )

    def learn(self, run):
        if len(self.memory) < self.memory.warmup:
            return
        
        # Sample a random minibatch of N transitions (si, ai, ri, si+1) from R
        state, action, reward, next_state, not_terminated = self.memory.sample(self.batch_size)

        state = state.to(self.actor.device)
        action = action.to(self.actor.device)
        reward = reward.to(self.actor.device)
        next_state = next_state.to(self.actor.device)
        not_terminated = not_terminated.to(self.actor.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        # Update the critic network
        target_actions = self.target_actor(next_state)
        target_q_value = self.target_critic(next_state, target_actions).squeeze(1)
        
        critic_q_value = self.critic(state, action)

        # Calculate y = ri + γQ'(si+1, μ'(si+1|θμ')|θQ')
        target = reward + self.gamma * torch.mul(target_q_value, not_terminated)

        
        target = torch.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = nn.functional.mse_loss(target, critic_q_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic(state, self.actor(state)).mean()
        actor_loss.backward()
        self.actor.optimizer.step()

        run.log({
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
        })

        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()
    

if __name__ == "__main__":
    # Test the OUActionNoise class
    ou_noise = OUActionNoise(mu = torch.zeros(1))
    noise = ou_noise()
    print(noise)

    # Test the ReplayBuffer class
    replay_buffer = ReplayBuffer(capacity = 120)
    for i in range(100):
        replay_buffer.add(
            state = torch.randn(3),
            action = torch.randn(1),
            reward = torch.randn(1),
            next_state = torch.randn(3),
            done = torch.tensor(0)
        )
    minibatch = replay_buffer.sample(batch_size = 10)
    if minibatch is not None:
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = minibatch
    else:
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = None, None, None, None, None
    print(state_batch)
    print(action_batch)
    print(reward_batch)
    print(next_state_batch)
    print(terminal_batch)

    print(len(replay_buffer))

    # Test the Agent class
    agent = Agent(
        lr_actor = 0.001,
        lr_critic = 0.001,
        state_dims = 8,
        gamma = 0.99,
        tau = 0.001,
        n_actions = 2,
        max_size = 1e6,
        warmup_percentage = 0.75,
        fc1_dims = 400,
        fc2_dims = 300,
        batch_size = 64
    )
    print(agent.actor)
    print(agent.critic)
    print(agent.target_actor)
    print(agent.target_critic)
    print(agent.choose_action(torch.randn(64, 8)))