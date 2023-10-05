import pygame
import random
from enum import Enum
import pandas as pd
import time

from collections import namedtuple
# namedtuple assigns a meaning to EACH position in a tuple
# Allows for a readableand self-documented code

pygame.init()
font = pygame.font.Font('arial.ttf', 25) # (font, font size)
font_gameOver = pygame.font.Font('Aloevera.ttf', 80)
font_gameOver2 = pygame.font.Font('Aloevera.ttf', 30)
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
RED = (225, 0, 0)
DARK_RED = (150, 0, 0)
DARK_BLUE = (0, 0, 255)
LIGHT_BLUE = (0, 100, 255)
LIGHTER_BLUE = (0, 120, 255)

class SnakeGame():
    def __init__(self, width = 640, height = 480) -> None:
        self.w = width # Values are in pixels
        self.h = height # Values are in pixels
        self.SPEED = 7.5 # Frame speed (Higher the number, faster the game is)
        
        self.display = pygame.display.set_mode((self.w, self.h))    # initializing display
        pygame.display.set_caption('Snake')                         # Setting caption
        self.clock = pygame.time.Clock()                            # Setting clock objest for SPEED

        # Initialize the game state
        self.direction = Direction.RIGHT                            # Initial direction
        
        # We do this and not use a string 
        # As we might mix up upper and lower case
        # E.g. maybe we initialised "right", but user gave "Right", then it won't work
        # Hence, we use a class Direction inheriting from the class Enum
 
        self.head = Point(self.w/2, self.h/2) # Coordinates of the initial location of the head of the snake
        
        ## self.head = [self.w/2, self.h/2] 
        # We are not using this as a list
        # As we have a better option of using `namedtuple`

        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y), Point(self.head.x - 2*BLOCK_SIZE, self.head.y)]
        # .x is used beacuse we initialised the `namedtuple` object `Point` that way
        # BLOCK_SIZE is the size of each snake block = 20 pixels
        # We have initialised the snake to be of 
        # # length = 3
        # # Head pointing in the right direction

        self.score = 0      # Initial score
        self.food = None    # Initialising Food location
        self._place_food()  # Initial food location --> Set randomly

    def _place_food(self): # Randomly set the coordinates of the food location
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake: # To ensure the food is NOt INSIDE the snake
            self._place_food()
    
    def play_step(self):
        # 1. Collect the user input
        for event in pygame.event.get(): # Gets all the user events in ONE play step
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN: # To check if user pressed any key
                if event.key == pygame.K_LEFT: # To check if the key is the LEFT key
                    if not self.direction == Direction.RIGHT: # If already moving in opposite direction, NO U-TURN allowed
                        self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT: # To check if the key is the RIGHT key
                    if not self.direction == Direction.LEFT:
                        self.direction = Direction.RIGHT
                elif event.key == pygame.K_DOWN: # To check if the key is the DOWN key
                    if not self.direction == Direction.UP:
                        self.direction = Direction.DOWN
                elif event.key == pygame.K_UP: # To check if the key is the UP key
                    if not self.direction == Direction.DOWN:
                        self.direction = Direction.UP

        # 2. Move the snake
        self._move(self.direction) # Will move the head of the snake in the new direction
        self.snake.insert(0, self.head) # Inserts the new head value at 0th position in the list
        # We don't use append as append will add it to the end and not in the start
        
        # 3. Check if game_over (Hit boundary or itself)
        game_over = False
        if self._is_collision():
            game_over = True
            text = font_gameOver.render("GAME OVER!", True, RED)
            text2 = font_gameOver2.render('Final Score = ' + str(self.score), True, DARK_RED)
            self.display.blit(text, [self.w//8,self.h//2-2*BLOCK_SIZE]) # Places the text on the center
            self.display.blit(text2, [self.w//3,self.h//2+2*BLOCK_SIZE]) # Places the text on the center


            # Now we need to update all these changes to the screen
            pygame.display.flip() # Updates the full display surface to the screen
            time.sleep(4)
            return game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score+=1
            self.SPEED+=0.75 # Increase the speed by 0.5 each time the snake successfully eats the food
            self._place_food() # Place the food randomly somewhere else
        else:
            self.snake.pop() # To remove the last element of the list

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(self.SPEED)

        # 6. Return if game over and score
        return game_over, self.score

    def _is_collision(self):
        # Hits boundary
        if ((self.head.x > self.w - BLOCK_SIZE) or (self.head.x < 0) or (self.head.y > self.h - BLOCK_SIZE) or (self.head.y < 0)):
            return True
        
        # Hits itself
        if self.head in self.snake[1:]: # Not checking with head
            return True
        
        return False

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT: # Can avoid the "right", "RIGHT", "Right" lower and upper case problems by using Enum class
            x+=BLOCK_SIZE
        elif direction == Direction.LEFT:
            x-=BLOCK_SIZE
        elif direction == Direction.UP:
            y-=BLOCK_SIZE # For up we subtract
        elif direction == Direction.DOWN:
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



if __name__ == '__main__':
    name = input('Name: ')
    game = SnakeGame()

    score_table = pd.read_csv('Score Table.csv', index_col=0)
    k=0
    loc = 0

    # Game loop
    while True:
        game_over, score = game.play_step()

        if game_over==True:
            break

        for names in score_table['Names']:
            if name == names:
                k=1
                break
            loc+=1
        if k==0:
            score_table_added = pd.DataFrame([[name, score]], columns=['Names', 'Best Score'])
    
    if k==0:
        score_table = pd.concat((score_table, score_table_added))
    else:
        if score_table['Best Score'][loc] < score:
            score_table['Best Score'][loc] = score
    score_table.to_csv('Score Table.csv')
    print(score_table)
    

    print("Final Score =",score)
    pygame.quit()
