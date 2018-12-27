'''This file contains the parameters determining the configuration of the RL setup.'''

## GENERAL PARAMETERS ##
SAVE_MEDIA = False # write out the media files: true/false

## DQN-RL parameters ###
EPISODES = 2000 # How many games to play
BATCH_SIZE = 32 # batch size used for reinforcing the algorithm
LEARNING_RATE = 0.001 # the learning rate for the CNN
EPSILON = 1.0 # the starting value for the curiosity parameter
EPSILON_MIN = 0.01 # the ending value for the curiosity parameter
EPSILON_DECAY = 0.999 # the decay for the parameter
GAMMA = 0.95 # weight for the reward of the next state given the action in the current state

## GAME parameters##
PIXELS_PER_TILE = 25 # Image size: How many pixels for one tile?

