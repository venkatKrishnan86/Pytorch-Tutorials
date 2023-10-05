import torch
from torch import nn
import torch.nn.functional as F
import os

class DeepQNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, action_size = 3) -> None:
        super(DeepQNetwork, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        # self.norm2 = nn.BatchNorm1d(hidden_size2)
        self.linear2 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, filename):
        model_folder_path = '../models' # Folder path
        if not os.path.exists(model_folder_path): # To check if the folder exists
            os.makedirs(model_folder_path) # If the folder does not exist, the create one
        
        filename = os.path.join(model_folder_path, filename) # Join the filename to the folder
        torch.save(self.state_dict(), filename) # Save the model as .pth file with the name --> filename


class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.lr = lr # Learning Rate
        self.gamma = gamma # Discount Rate
        self.model = model # The DQN model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss() # Aim to minimise (Q_new - Q_old)**2 --> Policy

    def train_step(self, state_old, action, reward, state_new, game_over):
        # We need to ensure we can implement batches of states, actions etc. 
        # As we have called this function with a batch of the parameters for training

        # All of these parameters can be the following -
        # - Tuples
        # - Lists 
        # - Single value
        # Hence we need to write the program accordingly to cater to all possibilities
        # game_over is a boolean value and doesn't have to be a torch tensor
        state_old = torch.tensor(state_old, dtype=torch.float32)
        state_new = torch.tensor(state_new, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long) # Long data type
        reward = torch.tensor(reward, dtype=torch.float32)
        # (n, x) --> Shape of these variable IF sent in a batch

        # Handling multiple sizes
        if len(state_old.shape) == 1:
            # x --> (1, x)
            state_old = torch.unsqueeze(state_old, 0) # Put 1 dimension in the 0th position in .shape
            state_new = torch.unsqueeze(state_new, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,) # Tuple with one value

        # 1. Predicted Q values with current state
        pred = self.model(state_old) # --> Gives 3 values

        # 2. Q_new = R + y * max(next_predicted Q value)
        # The max operation gives only 1 value
        # Hence we shall use pred.clone() --> To get 3 values
        # preds[argmax(action)] = Q_new
        target = pred.clone()
        for idx in range(len(game_over)): # Over the length of the batch of games played
            Q_new = reward[idx] # Reward of idx^th game in the batch
            if not game_over[idx]:
                # BELLMAN EQUATION
                Q_new = reward[idx] + self.gamma * torch.max(self.model(state_new[idx]))
            target[idx][torch.argmax(action).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()





        


        




