import torch
import random
import numpy as np
from collections import deque # Dequeue data structure
from snake_game import *
from DQN_better_states import DeepQNetwork, QTrainer
from plot_helper import plot_all_scores

MAX_MEMORY = 200_000
BATCH_SIZE = 5000
LR = 0.001

class Agent:
    def __init__(self) -> None:
        self.num_games = 0 # Number of games played
        self.epsilon = 0 # Parameter to control the randomness
        self.gamma = 0.9 # Discount rate (General value is around 0.8-0.9, always MUST be < 1)
        self.memory = deque(maxlen=MAX_MEMORY) # if crosses the memory it will delete the last elements (popleft())

        self.model = DeepQNetwork(15, 512, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    def get_state(self, game): # Get the state defining the environment
        # 12 variables/dimensions
        # First 3: Determine if there is a danger immediately ahead, right or left
        # Next 4: Dangers in diagonals
        # Next 4: Current Direction where the snake is moving
        # Next 4: Location of the food with respect to the snake
        # Last 1: Length of snake

        head = game.snake[0] # First element of the snake list is the head

        # Immediate next points around the SNAKE's HEAD in all directions
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE) # y axis is positive downwards
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        # point_ld = Point(head.x - BLOCK_SIZE, head.y + BLOCK_SIZE)
        # point_lu = Point(head.x - BLOCK_SIZE, head.y + BLOCK_SIZE)
        # point_rd = Point(head.x + BLOCK_SIZE, head.y - BLOCK_SIZE) # y axis is positive downwards
        # point_ru = Point(head.x + BLOCK_SIZE, head.y + BLOCK_SIZE)

        # Gets the 4 state values determining the current direction of the snake
        dir_l = (game.direction == Direction.LEFT)
        dir_r = (game.direction == Direction.RIGHT)
        dir_u = (game.direction == Direction.UP)
        dir_d = (game.direction == Direction.DOWN)

        # food_dis = np.sqrt((head.x - game.food.x)**2 - (head.y - game.food.y)**2)

        var = head

        count = 1
        dis_l = 0
        if dir_l:
            while True:
                var = Point(head.x - count*BLOCK_SIZE, head.y)
                if game.is_collision(var):
                    dis_l = count
                    break
                count+=1

        count = 1
        dis_r = 0
        if dir_r:
            while True:
                var = Point(head.x + count*BLOCK_SIZE, head.y)
                if game.is_collision(var):
                    dis_r = count
                    break
                count+=1

        count = 1
        dis_u = 0
        if dir_u:
            while True:
                var = Point(head.x, head.y - count*BLOCK_SIZE)
                if game.is_collision(var):
                    dis_u = count
                    break
                count+=1
        
        count = 1
        dis_d = 0
        if dir_d:
            while True:
                var = Point(head.x, head.y + count*BLOCK_SIZE)
                if game.is_collision(var):
                    dis_d = count
                    break
                count+=1
            


        # dir_tail_l = 0
        # dir_tail_r = 0
        # dir_tail_u = 0
        # dir_tail_d = 0

        # pos_last = game.snake[-1]
        # pos_second_last = game.snake[-2]
        # if pos_last.x - pos_second_last.x == BLOCK_SIZE:
        #     dir_tail_l = 1
        # elif pos_last.x - pos_second_last.x == -BLOCK_SIZE:
        #     dir_tail_r = 1
        # elif pos_last.y - pos_second_last.y == BLOCK_SIZE:
        #     dir_tail_u = 1
        # elif pos_last.y - pos_second_last.y == -BLOCK_SIZE:
        #     dir_tail_d = 1

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)),

            # # Danger straight-left diagonally
            # (dir_r and game.is_collision(point_ru)) or
            # (dir_d and game.is_collision(point_rd)) or
            # (dir_l and game.is_collision(point_ld)) or
            # (dir_u and game.is_collision(point_lu)),

            # #  Danger straight-right diagonally
            # (dir_r and game.is_collision(point_rd)) or
            # (dir_d and game.is_collision(point_ld)) or
            # (dir_l and game.is_collision(point_lu)) or
            # (dir_u and game.is_collision(point_ru)),

            # #  Danger down-right diagonally
            # (dir_r and game.is_collision(point_ld)) or
            # (dir_d and game.is_collision(point_lu)) or
            # (dir_l and game.is_collision(point_ru)) or
            # (dir_u and game.is_collision(point_rd)),

            # #  Danger down-left diagonally
            # (dir_r and game.is_collision(point_lu)) or
            # (dir_d and game.is_collision(point_ru)) or
            # (dir_l and game.is_collision(point_rd)) or
            # (dir_u and game.is_collision(point_ld)),

            # Direction of snake (4 state values)
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Tail Direction
            # dir_tail_l,
            # dir_tail_r,
            # dir_tail_u,
            # dir_tail_d,

            # Food location
            game.food.x < game.head.x, # Food is on the LEFT
            game.food.x > game.head.x, # Food is on the RIGHT
            game.food.y < game.head.y, # Food is UP
            game.food.y > game.head.y, # Food is DOWN

            # Food Distance
            # food_dis,

            # Distance of Danger
            dis_l,
            dis_r,
            dis_u,
            dis_d
        ] 

        return np.array(state, dtype = int) # int used to convert boolean to int

    def remember(self, state, action, reward, next_state, game_over):
        # Store the data to memory dequeue as a tuple of 5 values
        self.memory.append((state, action, reward, next_state, game_over)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        # Create a batch size of 1000 from our self.memory variable
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
            # Randomly chooses BATCH_SIZE number of previous memories
        else:
            mini_sample = self.memory

        # All the states together, actions together etc.
        states, actions, rewards, next_states, game_overs = zip(*mini_sample)
        # OR can do it this way (which would be a bit slower)
        ## states = []
        ## action = []
        ## for state, action, reward, next_state, game_over in mini_samples:
        ##     states.append(state)
        ##     actions.append(action)
        # and so on...

        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, state, action, reward, next_state, game_over):
        # Train over ONE game step
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state): # Given state, predict action using Deep Learning
        # RANDOM MOVES in the beginning: tradoff between exploration & exploitation
        # We want the agent to explore the environment in the beginning
        # Slowly, it must train and exploit it to a way that it completely learns how to beat the game

        # Explore, and then exploit after a certain number of games (Here it is 80)
        self.epsilon = 80 - self.num_games # Measure of randomness --> Decreases with the number of games
        final_move = [0,0,0] # Action: (Straight, Right, Left)

        if random.randint(0, 200) < self.epsilon: # If true --> Perform random move
            # Starts with a 40% chance of randomly choosing an action for exploration 
            # (Since epsilon = 80, when num_games = 0)
            move = random.randint(0,2)
            final_move[move] = 1 # Randomly go straight, right or left so that the model can freely explore
        else:
            # Move based on our model
            state0 = torch.tensor(state, dtype = torch.float32) # Although state vector is int, float is used since we need to input this into a Neural Network
            prediction = self.model(state0) # Predict the action based on the current state (Calls the forward() function)

            # Now we will find the max value position and convert that to 1, rest as 0 to get a move
            move_idx = torch.argmax(prediction).item() # returns an integer index of the max value location (argmax)
            final_move[move_idx] = 1
        
        return final_move

def train():
    plot_scores = [] # Scores with games
    plot_mean_scores = [] # Average scores
    total_score = 0 # Total score for 1 game
    mean_score = 0
    record = 0 # Best score in game

    agent = Agent() # The snake we are going to train
    game = SnakeGameAI()
    
    while True:
        # Get the old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # perform the move and get the new state
        reward, game_over, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over)

        # Store all the values for future use
        agent.remember(state_old, final_move, reward, state_new, game_over)

        if game_over:
            # Train the long memory/replay memory - tremendously helps the agent improve the result
            game._reset() # Resetting the game score, snake etc. to start again
            agent.num_games +=1
            agent.train_long_memory()

            if score > record: # New high score
                record = score
                agent.model.save(filename='RL_Snake_Final.pth')
            
            print("Game",agent.num_games,"- Score:",score,"Record:",record)

            # TODO: Plot the result
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.num_games
            plot_mean_scores.append(mean_score)
            plot_all_scores(plot_scores, plot_mean_scores)




if __name__ == "__main__":
    train()

