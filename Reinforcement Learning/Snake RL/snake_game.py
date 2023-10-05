import pygame
import random
from enum import Enum
import numpy as np

from collections import namedtuple
# namedtuple assigns a meaning to EACH position in a tuple
# Allows for a readableand self-documented code

pygame.init()
font = pygame.font.Font('arial.ttf', 25) # (font, font size)
# Using System Font (This makes the game loading SLOWER)
## font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4 # No lower case or mix upper and lower case problems

Point = namedtuple('Point', 'x, y') # Defining a namedtuple for defining locations in the game
# First parameter - is same as the variable name
# Second parameter - comma separated data values

BLOCK_SIZE = 20 # We want each part of the snake to be of 20 PIXELS SIZE

# Colours (RGB values) (Ranges from 0 - 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
DARK_BLUE = (0, 0, 255)
LIGHT_BLUE = (0, 100, 255)
LIGHTER_BLUE = (0, 120, 255)

SPEED = 40 # Frame speed (Higher the number, faster the game is)

class SnakeGameAI():
    def __init__(self, width = 640, height = 480) -> None:
        self.w = width # Values are in pixels
        self.h = height # Values are in pixels
        
        self.display = pygame.display.set_mode((self.w, self.h))    # initializing display
        pygame.display.set_caption('Snake')                         # Setting caption
        self.clock = pygame.time.Clock()                            # Setting clock objest for SPEED
        self._reset()
    
    def _reset(self):
        # Initialize the game state
        self.direction = Direction.RIGHT                            # Initial direction
        self.head = Point(self.w/2, self.h/2) # Coordinates of the initial location of the head of the snake
        self.snake = [
            self.head, 
            Point(self.head.x - BLOCK_SIZE, self.head.y), 
            Point(self.head.x - 2*BLOCK_SIZE, self.head.y)
        ]
        self.score = 0      # Initial score
        self.food = None    # Initialising Food location
        self._place_food()  # Initial food location --> Set randomly

        self.frame_iteration = 0 # Keeping track of the game/frame iteration

    def _place_food(self): # Randomly set the coordinates of the food location
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake: # To ensure the food is NOt INSIDE the snake
            self._place_food()
    
    def play_step(self, action): # Action from the agent
        self.frame_iteration+=1
        # 1. Collect the user input
        for event in pygame.event.get(): # Gets all the user events in ONE play step
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move the snake
        self._move(action) # Will move the head of the snake in the new direction
        self.snake.insert(0, self.head) # Inserts the new head value at 0th position in the list
        # We don't use append as append will add it to the end and not in the start
        
        # 3. Check if game_over (Hit boundary or itself)
        reward = 0 # Reward for the agent: Eat food: +10, Hit boundary: -10, else 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake): 
            # frame_iteration > 100*len(self.snake): If the snake just roams around for too long without eating the food, then penalise it
            # Longer the snake, it has more time
            reward = -10
            game_over = True
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score+=1
            reward = 10
            self._place_food() # Place the food randomly somewhere else
        else:
            self.snake.pop() # To remove the last element of the list

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. Return if game over and score
        return reward, game_over, self.score

    # Underscore before the function means it is a private function (def _is_collision())
    # Removing it makes it public
    def is_collision(self, pt = None): # Using pt to check the danger values in the state vector
        if pt is None:
            pt = self.head
        # Hits boundary
        if ((pt.x > self.w - BLOCK_SIZE) or (pt.x < 0) or (pt.y > self.h - BLOCK_SIZE) or (pt.y < 0)):
            return True
        
        # Hits itself
        if pt in self.snake[1:]: # Not checking with head
            return True
        
        return False

    def _move(self, action): # Action has 3 values
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP] # In clockwise order
        idx = clock_wise.index(self.direction) # The index value of the current direction the snake is moving in

        if np.array_equal(action, [1,0,0]): # Straight
            new_dir = clock_wise[idx] # Current direction
        elif np.array_equal(action, [0,1,0]): # Right from current direction
            next_idx = (idx+1)%4 # Next index, %4 if we are going from up to right
            new_dir = clock_wise[next_idx] # Right turn: r -> d -> l -> u -> r
        elif np.array_equal(action, [0,0,1]): # Left from current direction
            next_idx = (idx-1)%4 # Previous index, %4 if we are going from up to right
            new_dir = clock_wise[next_idx] # Left turn: r -> u -> l -> d -> r (Anti-clockwise)

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x+=BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x-=BLOCK_SIZE
        elif self.direction == Direction.UP:
            y-=BLOCK_SIZE # For up we subtract
        elif self.direction == Direction.DOWN:
            y+=BLOCK_SIZE

        self.head = Point(x,y)


    def _update_ui(self):
        self.display.fill(BLACK)
        for i, pt in enumerate(self.snake):
            # The main blocks of the snake (Size 20 x 20)
            pygame.draw.rect(self.display, DARK_BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # Inside block to distinguish each block in the snake (Size 12 x 12)
            if i!=0:
                pygame.draw.rect(self.display, LIGHT_BLUE, pygame.Rect(pt.x+4, pt.y+4, BLOCK_SIZE-8, BLOCK_SIZE-8))
            else:
                pygame.draw.rect(self.display, LIGHTER_BLUE, pygame.Rect(pt.x+4, pt.y+4, BLOCK_SIZE-8, BLOCK_SIZE-8))
        
        # Drawing the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # pygame.draw.circle(self.display, RED, (self.food.x, self.food.y), BLOCK_SIZE//2)

        text = font.render("Score: "+str(self.score), True, WHITE)
        self.display.blit(text, [0,0]) # Places the text on the upper left part of the display

        # Now we need to update all these changes to the screen
        pygame.display.flip() # Updates the full display surface to the screen
