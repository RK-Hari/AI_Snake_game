import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('FFF_Tusj.ttf', 25)

#Convert the directions to numbers
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Color and variables
SKY_BLUE = (176, 225, 255)
LIGHT_BROWN = (218, 180, 143) 
CRIMSON = (221, 81, 127)
ORANGE = (230, 142, 54)
DARK_PURPLE = (70, 30, 82)

#Size of the parts in the screen and the speed of the snake
BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    
    #Initialising all values used
    def __init__(self, w=840, h=680):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('AI Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
    #Resetting the values of the games, when a collision occurs
    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()
        
    #This function places food randomly in the game window
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    #This function is used to find whether collision has occured or not
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True

        return False
    
    #To set up the UI of the game
    def _update_ui(self):
        self.display.fill(DARK_PURPLE)
    
        for pt in self.snake:
            pygame.draw.circle(self.display, CRIMSON, (pt.x + BLOCK_SIZE // 2, pt.y + BLOCK_SIZE // 2), BLOCK_SIZE // 2)
            pygame.draw.circle(self.display, ORANGE, (pt.x + BLOCK_SIZE // 2, pt.y + BLOCK_SIZE // 2), 4)
    
        pygame.draw.rect(self.display, LIGHT_BROWN, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
    
        text = font.render("Score: " + str(self.score), True, SKY_BLUE)
        text_rect = text.get_rect(center=(self.w // 2, 20))
        self.display.blit(text, text_rect)
        pygame.display.flip()

    #To determine the direction to move 
    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        #If same direction, then no change in direction
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] 
        #If different direction then , predict the next move as down to teach the AI with directions 
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)