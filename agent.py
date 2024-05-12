import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

#Max memory of the deque
MAX_MEMORY = 100_000
#Once at a time
BATCH_SIZE = 1000
#Learning Rate
#The learning rate determines the step size at which the model's parameters are updated during training
LR = 0.001

class Agent:

    def __init__(self):
        #Number of games the AI has completed
        self.n_games = 0
        #For randomness
        self.epsilon = 0 
        #Discount_rate
        #It quantifies the extent to which the model values immediate rewards over future rewards.
        self.gamma = 0.9 
        #To store the actions performed and if the size exceeds the specified size, then it pops elements from the left
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        #Getting the head of the snake
        head = game.snake[0]
        #Calculating one point ahead the snake's head in all directions
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        #Determining the current direction
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #Calculating the danger direction
            # Danger is straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger is at the right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger is at the left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]
        
        #Whole structure a.k.a neural network
        #[0 , 0 , 0
        # 0 , 0 , 1 , 0
        # 1 , 0 , 1 , 0] 
        
        #Returning the neural structure in array format , after casting it into "int" type
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        #Appending the action performed, along with the reward for learning
        self.memory.append((state, action, reward, next_state, done)) 
        # It automatically pops left if MAX_MEMORY is reached
        
    #We use long memory for storing 1000 samples(BATCH_SIZE) from the memory, for effient training of the AI.
        
    #It enables efficient training and also parallelisation.
    #Efficient Memory Usage: Training a neural network on a large dataset can be memory-intensive. 
    #By training the network on smaller batches of data at a time, you can reduce memory usage. 
    #This is especially important if you're working with limited memory resources, such as GPUs.
    #Parallelization: Many deep learning frameworks and hardware accelerators (like GPUs) are optimized for parallel computation. 
    #By training on batches of data, you can take advantage of parallelization to speed up the training process.
    def train_long_memory(self):
        #IF size is larger, randomly take 1000(BATCH_SIZE) samples from memory and store it here.
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        #If size is smaller, then store it in memory.
        else:
            mini_sample = self.memory
            
        #zip(*mini_sample) expression is used to unzip 
        #the list of tuples mini_sample into separate lists for each element of the tuple
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        #You can also use:
        #for state,action,reward,next_state,done in mini_sample:
        #    self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    #Short memory for storing every single game    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    #This function is used to determine random moves to perform
    #Exploration helps in discovering new states and actions, which is essential for learning an optimal policy.
    #(without considering their predicted values)
    #Exploitation leverages the learned knowledge to maximize rewards in the short term.
    #(make decisions that are expected to yield higher rewards)
    def get_action(self, state):
        # epsilon (ε) is commonly used to balance exploration and exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        #Exploration
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        #Exploitation
        else:
            # Converting the state into tensor format allows it to be processed by the model.
            # A tensor is a multi-dimensional array
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            # The action with the highest predicted Q-value is selected using argmax. 
            # The item() method is used to extract the integer value of the selected action.
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    #For plotting using matplotlib
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # getting the old(current) state
        state_old = agent.get_state(game)

        # getting the move based on the current state
        final_move = agent.get_action(state_old)

        # perform the move and save the values, play_step is in game.py file
        reward, done, score = game.play_step(final_move)
        # getting the new updated state of the snake
        state_new = agent.get_state(game)

        # training short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember all the values for training purposes
        agent.remember(state_old, final_move, reward, state_new, done)
        
        #If collision occured
        if done:
            # Train the long memory and plot it in graph
            #Reset the game
            game.reset()
            #Increment the number of the games
            agent.n_games += 1
            #Train the long memory
            agent.train_long_memory()
            
            #Check for new high score
            if score > record:
                record = score
                agent.model.save()
                
            #Print the number of games,score and high_score after each iteration
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
            #Adding the score to the plot_scores array
            plot_scores.append(score)
            #Add it to the total score , to calculate the mean
            total_score += score
            #Calculate the mean
            mean_score = total_score / agent.n_games
            #Append the mean score to the mean score list
            plot_mean_scores.append(mean_score)
            #Plot it in the graph
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()