import torch
import random
import numpy as np
from collections import deque  # use to store data
from snake import SnakeAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot
import os
import sys


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
SPEED = 20000  # Increase the frame rate
INPUT_SIZE = 12
HIDDEN_SIZE = 255
OUTPUT_SIZE = 3

ARGS = sys.argv

class Agent:
    """
    The Agent class represents an agent for the SnakeAI Game.
    
    Attributes:
        n_games (int): The number of games played by the agent.
        epsilon (float): The randomness factor for exploration.
        gamma (float): The discount rate for future rewards.
        memory (deque): A deque to store game data. (deque is a list-like container with fast appends and pops on either end)
        model (object): The model used for prediction.
        trainer (object): The trainer used for training the model.
    Methods:
        get_state(game): Returns the current state of the game.
        remember(state, action, reward, next_state, done): Stores game data in memory.
        train_long_memory(): Trains the model using long-term memory.
        train_short_memory(state, action, reward, next_state, done): Trains the model using short-term memory.
        get_action(state): Returns the action to take based on the current state.
    """

    def __init__(self):
        self.use_old = False
        self.n_games = 0
        self.epsilon = 0  
        self.gamma = 0.9  
        self.memory = deque(maxlen=MAX_MEMORY)  # save data to deque
        self.model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)  # 11 input, 256 hidden, 3 output

        if os.path.exists("./model/model.pth") and len(ARGS) == 2 and str(ARGS[1]) == 'load':
            self.model.load_state_dict(torch.load('./model/model.pth'))
            self.model.eval()
            self.use_old = True

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # QTrainer object

    def get_state(self, game):
        """
        Returns the state of the game for the agent in 0 and 1 format.
        
        Parameters:
            game (Game): The game object representing the current state of the game.
        Returns:
            state (numpy.ndarray): An array representing the state of the game. The array contains the following elements:
                Danger Direction: True if there is a danger straight/right/left/below ahead in the current direction, False otherwise
                Food Location: True if the food is to the left/right/above/below of the snake's head, False otherwise.
        """

        head = game.snake[0]
        point_r = Point(head.x + 20, head.y)
        point_l = Point(head.x - 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_r = game.direction == Direction.RIGHT
        dir_l = game.direction == Direction.LEFT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            # Ex: True if there is a danger to the right in the current direction, False otherwise
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Danger down
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location (start from head location)
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y > game.head.y,  # food up
            game.food.y < game.head.y,  # food down

            # Potential traps (future collision if snake moves in this direction)
            # game.is_collision(Point(head.x + 2 * BLOCK_SIZE, head.y)) if dir_r else 0,
            # game.is_collision(Point(head.x - 2 * BLOCK_SIZE, head.y)) if dir_l else 0,
            # game.is_collision(Point(head.x, head.y - 2 * BLOCK_SIZE)) if dir_u else 0,
            # game.is_collision(Point(head.x, head.y + 2 * BLOCK_SIZE)) if dir_d else 0
        ]

        return np.array(state, dtype=int)  # Turn State (True or False boolean) into 0 and 1

    def remember(self, state, action, reward, next_state, done):
        # append a list of tuple
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    """
    Train_long_memory & Train short memory

    Args:
        state (np.array): The current state of the game.
        action (int): The action taken.
        reward (float): The reward received.
        next_state (np.array): The next state of the game.
        done (bool): Indicates if the game is done.

    Returns:
        the action to take based on the current state.
    """

    def train_long_memory(self):
        """
        This method is responsible for training the agent using the samples stored in the long-term memory.
        If the number of samples in the memory is greater than the batch size, a mini-batch of size BATCH_SIZE is randomly selected.
        Otherwise, all the samples i n the memory are used.
        """
        if len(self.memory) > BATCH_SIZE:  # if have more than BATCH_SIZE samples
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)  # Extract every samples existed
        self.trainer.train_step(states, actions, rewards, next_states, dones)  # Train with every samples existed

    def train_short_memory(self, state, action, reward, next_state, done):  # 1 step
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Returns a one-hot encoded list representing the action to take based on the given state.
        
        Parameters:
        - state: The current state of the game.
        Returns:
        - final_move: A one-hot encoded list [0, 0, 0] where only one element is 1, indicating the direction to move.
        Explain:
            Random moves: tradeoff exploration / exploitation
            if random < epsilon: explore 
            else: exploit
        """

        self.epsilon = 69 - self.n_games  # epsilon get smaller as the game progress

        final_move = [0, 0, 0]

        if not self.use_old and random.randint(0, 200) < self.epsilon:
            # self.epsilon = 80 - self.n_games
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

            # Prevent moves leading to dead ends
            # if state[0] and move == 0:  # Danger straight
            #     move = random.choice([1, 2])
            # elif state[1] and move == 1:  # Danger right
            #     move = random.choice([0, 2])
            # elif state[2] and move == 2:  # Danger left
            #     move = random.choice([0, 1])

            final_move[move] = 1

        return final_move
    
    


def train():
    """
    Trains the snake AI agent by playing the snake game.
    This function iteratively plays the snake game and trains the agent's neural network model.
    It keeps track of the scores and records, and updates the agent's memory for training.
    After each game, it resets the snake game and trains the agent's long-term memory.
    """

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    snake = SnakeAI()

    while True:
        # get old state
        state_old = agent.get_state(snake)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = snake.play(final_move)
        state_new = agent.get_state(snake)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory. replay memory
            snake.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_scores = total_score / agent.n_games
            plot_mean_scores.append(mean_scores)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    # Create an instance of your model
    model = Linear_QNet(12, 255, 3)

    # Load the state dictionary from the file
    # model.load_state_dict(torch.load('./model/model.pth'))

    # Set the model to evaluation mode (optional, but recommended for inference)
    model.eval()

    train()
