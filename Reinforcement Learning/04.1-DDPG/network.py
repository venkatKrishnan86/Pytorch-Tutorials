# We shall be implementing the Actor and Critic networks here
import torch
from torch import nn
import numpy as np
import os

class Actor(nn.Module):
    def __init__(
        self,
        lr,
        input_dims,
        fc1_dims,
        fc2_dims,
        n_actions = 2,
        name = 'actor',
        chkpt_dir='tmp/ddpg'
    ):
        super(Actor, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg.pth')

        # LAYER 1
        # -------
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        # Initialize the weights and biases of the first fully connected layer
        # Constraining the weights to be within a range of [-f1, f1]
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        # LAYER 2
        # -------
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        # Initialize the weights and biases of the second fully connected layer
        # Constraining the weights to be within a range of [-f2, f2]
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # OUTPUT LAYER
        # ------------
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        f3 = 0.003
        torch.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """
            Arguments:
            ----------
            state: torch.Tensor
                The input state tensor

            Returns:
            --------
            action: torch.Tensor
                The output action tensor (deterministic)
        """
        x = torch.relu(self.bn1(self.fc1(state)))
        x = torch.relu(self.bn2(self.fc2(x)))
        action = torch.tanh(self.mu(x))

        return action
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(torch.load(self.checkpoint_file))


class Critic(nn.Module):
    def __init__(
        self,
        lr,
        input_dims,
        fc1_dims,
        fc2_dims,
        n_actions,
        name,
        chkpt_dir='tmp/ddpg'
    ):
        super(Critic, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg.pth')

        # LAYER 1
        # -------
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        # Initialize the weights and biases of the first fully connected layer
        # Constraining the weights to be within a range of [-f1, f1]
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        # LAYER 2
        # -------
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        # Initialize the weights and biases of the second fully connected layer
        # Constraining the weights to be within a range of [-f2, f2]
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        f3 = 3e-3
        self.q = nn.Linear(self.fc2_dims, 1)
        torch.nn.init.uniform_(self.q.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        """
            Arguments:
            ----------
            state: torch.Tensor
                The input state tensor
            action: torch.Tensor
                - The input action tensor
                - Actor network's output
                - Inputted in the second layer of the critic network

            Returns:
            --------
            q_value: torch.Tensor
                The output Q-value tensor
        """
        state_value = torch.relu(self.bn1(self.fc1(state)))
        state_value = self.bn2(self.fc2(state_value))
        action_value = torch.relu(self.action_value(action))
        state_action_value = torch.relu(state_value + action_value)
        q_value = self.q(state_action_value)

        return q_value
    
    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))