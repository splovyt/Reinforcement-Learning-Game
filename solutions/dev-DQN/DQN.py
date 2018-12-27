'''This file contains the algorithm for the DQN reinforcement learning agent.'''

# source files
from settings import *

# standard libraries
from collections import deque
import random

# additional libraries
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers import Dropout
from keras.layers import Flatten


class DQN_RL:

    def __init__(self):
        '''Initialize the DQN RL agent.'''
        self.q_table = deque(maxlen=5000)  # i.e. the "training" data
        self.epsilon = EPSILON # the curiosity parameter (i.e. fraction of random action sampling)

    def initialize_actions(self, actions_dict):
        '''Initialize the action space of the agent, i.e. all possible actions.'''
        # action name to function mapping
        self.actions_dict = actions_dict
        # action names
        self.actions = sorted(list(actions_dict.keys()))
        # amount of actions
        self.n_actions = len(self.actions)
        # give number to each action
        self.int2actions = {self.actions.index(x): x for x in self.actions}
        self.actions2int = {x: self.actions.index(x) for x in self.actions}

    def buildCNN(self, game_obj, input_img_height, input_img_width):
        '''Build the CNN to be able to read the state of the game the same way the user can,
        i.e. visually.'''
        # initialize a sequential model
        self.model = Sequential()

        # attempt to shape one tile as one kernel
        kernel_size = (int(input_img_width / game_obj.board.shape[1]), # kernel width
                       int(input_img_height / game_obj.board.shape[0])) # kernel height
        pixels_in_image = game_obj.board.shape[0] * game_obj.board.shape[1]

        self.model.add(Conv2D(pixels_in_image, kernel_size, input_shape=(input_img_height, input_img_width, 1),
                              strides=int(input_img_width / game_obj.board.shape[1]), activation='relu'))

        # attempt to shape a combination of tiles as a kernel
        self.model.add(Conv2D(int(pixels_in_image // 4), kernel_size=(5, 5),
                              input_shape=(input_img_height, input_img_width, 1),
                              activation='relu'))

        # flatten the convolution layers
        self.model.add(Flatten())
        # dense layer
        self.model.add(Dense(50, activation='relu'))
        # dropout layers to prevent overfitting
        self.model.add(Dropout(0.5))
        # dense layer
        self.model.add(Dense(28, activation='relu'))
        # Output layer representing each action
        self.model.add(Dense(self.n_actions, activation='linear'))
        # compile the model
        self.model.compile(loss='mse',
                           optimizer=Adam(lr=LEARNING_RATE))

    def act(self, state, epsilon=None):
        '''Generate the action given the current game state,
        while applying a curiosity parameter (gamma).'''
        if epsilon is None:
            epsilon = self.epsilon
        # the exploration vs. exploitation tradeoff
        if np.random.rand() <= epsilon:
            # exploration (= curiosity)
            action = random.choice(self.actions)
            action_func = self.actions_dict[action]
            # returns the action function
            return action_func
        # exploitation
        act_values = self.model.predict(state)
        action = self.int2actions[np.argmax(act_values[0])]
        action_func = self.actions_dict[action]
        # returns the action function
        return action_func

    def save_state(self, last_state, action, reward, new_state, done):
        '''Add the last state, the action, and its effect (reward, new state,
        and whether the game is over) to the Q-table.'''
        self.q_table.append((last_state, action, reward, new_state, done))

    def reinforce(self):
        '''Apply the core reinforcement learning algorithm to the collected data.'''
        # reinforce based on randomly sampled data
        minibatch = random.sample(self.q_table, BATCH_SIZE)
        for last_state, action, reward, new_state, done in minibatch:
            if not done:
                # Bellman equation for the Q function (only when game is in progress)
                reward = (reward + GAMMA * np.amax(self.model.predict(new_state)[0]))
            # request the predicted rewards
            predicted_rewards = self.model.predict(last_state)
            # adjust the predicted rewards with the known actual reward
            predicted_rewards[0][self.actions2int[action]] = reward
            # fit the model with the adjusted rewards (~ until convergence)
            self.model.fit(last_state, predicted_rewards, epochs=1, verbose=0)

        # decrease the curiosity parameter as we have seen more data
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY



