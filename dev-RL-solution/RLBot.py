## General ##
# write out the media files?
SAVE_MEDIA = False

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
GAMMA = 0.95

# Image size: How many pixels for one tile?
PIXELS_PER_TILE = 25

# How many games to play
EPISODES = 2000

from collections import deque

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers import Dropout
from keras.layers import Flatten


class DQN_RL:

    def __init__(self):
        '''Initialize the RL agent.'''
        self.q_table = deque(maxlen=5000)  # i.e. the "training" data
        self.epsilon = EPSILON

    def initialize_actions(self, actions_dict):
        # action name to function mapping
        self.actions_dict = actions_dict
        # action names
        self.actions = sorted(list(actions_dict.keys()))
        # amount of actions
        self.n_actions = len(self.actions)
        # give number to each action
        self.int2actions = {self.actions.index(x): x for x in self.actions}
        self.actions2int = {x: self.actions.index(x) for x in self.actions}

    def buildCNN(self, input_img_height, input_img_width):
        kernel_size = (int(input_img_width / game.board.shape[1]), int(input_img_height / game.board.shape[0]))

        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size, input_shape=(input_img_height, input_img_width, 1), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(50, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(28, activation='relu'))
        # Output layer representing each action
        self.model.add(Dense(self.n_actions, activation='linear'))
        self.model.compile(loss='mse',
                           optimizer=Adam(lr=LEARNING_RATE))

    def act(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        # the exploration vs. exploitation tradeoff
        if np.random.rand() <= epsilon:
            # exploration
            action = random.choice(self.actions)
            action_func = self.actions_dict[action]
            return action_func
        # exploitation
        act_values = self.model.predict(state)
        action = self.int2actions[np.argmax(act_values[0])]
        #print('predicted action', action, epsilon)
        action_func = self.actions_dict[action]
        return action_func  # returns action function

    def save_state(self, last_state, action, reward, new_state, done):
        '''Add the state, the action, and its effect (reward, next_state,
        and whether the game is over) to the Q-table. '''
        self.q_table.append((last_state, action, reward, new_state, done))

    def replay(self):
        minibatch = random.sample(self.q_table, BATCH_SIZE)
        for last_state, action, reward, new_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + GAMMA *
                          np.amax(self.model.predict(new_state)[0]))
            target_f = self.model.predict(last_state)
            target_f[0][self.actions2int[action]] = target
            self.model.fit(last_state, target_f, epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY


from PIL import Image


def rgb2gray(rgb):
    '''Convert an RGB-array to grayscale.'''
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def reshape_state(img, new_height, new_width):
    '''Reshape the state to have the correct input properties for the neural network.'''
    # resize image
    resized_img = np.array(Image.fromarray(img).resize((new_width, new_height)))
    # make grayscale
    resized_gray = rgb2gray(resized_img)
    # normalize
    resized_gray_normalized = resized_gray / 255.0
    # reshape for the network
    reshaped = resized_gray_normalized.reshape(1, new_height, new_width, 1)
    return reshaped


def reward_function(bombs_placed, terrain_added, win_lose_ongoing):
    # if the game ended
    if win_lose_ongoing == 'win':
        return 100
    elif win_lose_ongoing == 'lose':  # draw is a loss too
        return -100

    # if nothing happened, place a penalty for duration of the game
    elif bombs_placed == 0 and terrain_added == 0:
        return -1

    # calculate the reward
    else:
        return bombs_placed * 10 + terrain_added * 10


def calculate_bombs_placed(json):
    return len([x for x in json['board_positions']['bombs_and_stage'] if x[2] == 1])


def calculate_terrain_added(json, last_json):
    # extract the frame
    frame = json['game_properties']['frame']

    # terrain added
    if frame == 1:
        previous_amount_of_terrain = len(json['board_positions']['land'])
    else:
        previous_amount_of_terrain = len(last_json['board_positions']['land'])

    terrain_added = len(json['board_positions']['land']) - previous_amount_of_terrain

    return terrain_added


def calculate_reward(json, last_json, win_lose_ongoing):
    # bombs placed this frame
    bombs_placed = calculate_bombs_placed(json)

    # terrain added this frame
    terrain_added = calculate_terrain_added(json, last_json)

    # reward in this state
    return reward_function(bombs_placed, terrain_added, win_lose_ongoing)


# source imports
from game import Game, Player
from render_tool import RenderTool, MapScheme

# additional library imports
import math
import random
import pandas as pd
import numpy as np

# dev-RL-solution imports
from utils import save_dict
from tqdm import tqdm

RL_performance = []

# initialize the agent
agent_player1 = DQN_RL()
agent_player2 = DQN_RL()

# start
for e in tqdm(range(EPISODES)):
    # choose the map
    # map = MapScheme().IBM
    map = MapScheme().standard

    # initialize the game
    game = Game(map, verbose=False)
    RT = RenderTool(game)

    # name the players
    player1 = Player(game, 'Sonic')
    player2 = Player(game, 'Knuckles')

    # initialize the action space
    agent_player1.initialize_actions(actions_dict={'Up': player1.Up,
                                                   'Down': player1.Down,
                                                   'Left': player1.Left,
                                                   'Right': player1.Right,
                                                   'Still': player1.Still,
                                                   'Bomb': player1.Bomb})
    agent_player2.initialize_actions(actions_dict={'Up': player2.Up,
                                                   'Down': player2.Down,
                                                   'Left': player2.Left,
                                                   'Right': player2.Right,
                                                   'Still': player2.Still,
                                                   'Bomb': player2.Bomb})

    # calculate the properties of the input images
    pixel_height_tile = math.ceil(math.sqrt(PIXELS_PER_TILE))
    pixel_width_tile = pixel_height_tile
    input_image_height = pixel_height_tile * game.board.shape[0]
    input_image_width = pixel_width_tile * game.board.shape[1]

    # build the CNN
    agent_player1.buildCNN(input_image_height, input_image_width)
    agent_player2.buildCNN(input_image_height, input_image_width)

    # start the game (frame 1)
    if game.start():
        # update the frame
        game_status_dict = game.get_status_dict()
        save_dict('data/{}/{}.pickle'.format(game.id, game.frame), game_status_dict)

        # render the frame
        frame = np.array(RT.render_current_frame(SAVE_MEDIA))

        # reshape the frame (= the state)
        new_state = reshape_state(frame, new_height=input_image_height, new_width=input_image_width)

        # cumulative reward
        cumulative_reward_player1 = 0
        cumulative_reward_player2 = 0

        # track performance
        total_terrain_added = 0
        total_bombs_placed = 0

    while game_status_dict['game_properties']['outcome'] == 'ongoing':
        # set the latest new state to the last state
        last_state = new_state
        last_game_status_dict = game_status_dict

        # select an action for the player
        move_player1 = agent_player1.act(last_state)
        move_player2 = agent_player2.act(last_state)

        move_player1()
        move_player2()

        # update the frame
        game.update_frame()
        game_status_dict = game.get_status_dict()
        save_dict('data/{}/{}.pickle'.format(game.id, game.frame), game_status_dict)

        # render the frame
        frame = np.array(RT.render_current_frame(SAVE_MEDIA))

        # reshape the frame (= the state)
        new_state = reshape_state(frame, new_height=input_image_height, new_width=input_image_width)

        # update the Q-table
        done = False
        if game.ended:
            done = True

        if game_status_dict['game_properties']['outcome'] == 'ongoing':
            reward_player1 = calculate_reward(json=game_status_dict, last_json=last_game_status_dict,
                                              win_lose_ongoing='ongoing')
            reward_player2 = calculate_reward(json=game_status_dict, last_json=last_game_status_dict,
                                              win_lose_ongoing='ongoing')
        elif game_status_dict['game_properties']['outcome'] == 'draw':
            reward_player1 = calculate_reward(json=game_status_dict, last_json=last_game_status_dict,
                                              win_lose_ongoing='lose')
            reward_player2 = calculate_reward(json=game_status_dict, last_json=last_game_status_dict,
                                              win_lose_ongoing='lose')
        elif game_status_dict['game_properties']['outcome'] == 'player_1':
            reward_player1 = calculate_reward(json=game_status_dict, last_json=last_game_status_dict,
                                              win_lose_ongoing='win')
            reward_player2 = calculate_reward(json=game_status_dict, last_json=last_game_status_dict,
                                              win_lose_ongoing='lose')
        elif game_status_dict['game_properties']['outcome'] == 'player_2':
            reward_player1 = calculate_reward(json=game_status_dict, last_json=last_game_status_dict,
                                              win_lose_ongoing='lose')
            reward_player2 = calculate_reward(json=game_status_dict, last_json=last_game_status_dict,
                                              win_lose_ongoing='win')

        agent_player1.save_state(last_state=last_state,
                                 action=move_player1.__name__,
                                 reward=reward_player1,
                                 new_state=new_state,
                                 done=done)
        agent_player2.save_state(last_state=last_state,
                                 action=move_player2.__name__,
                                 reward=reward_player2,
                                 new_state=new_state,
                                 done=done)

        cumulative_reward_player1 += reward_player1
        cumulative_reward_player2 += reward_player2

        # track performance
        total_terrain_added += calculate_terrain_added(json=game_status_dict, last_json=last_game_status_dict)
        total_bombs_placed += calculate_bombs_placed(json=game_status_dict)

    if len(agent_player1.q_table) > BATCH_SIZE:
        agent_player1.replay()
        agent_player2.replay()

    # track performance
    RL_performance.append([agent_player1.epsilon, cumulative_reward_player1, cumulative_reward_player2,
                           game_status_dict['game_properties']['outcome'], len(game.players[0].history),
                           total_terrain_added, total_bombs_placed])

    if e > 0 and e % 10 == 0:
        pd.DataFrame(RL_performance, columns=['epsilon', 'cum_reward_p1', 'cum_reward_p2',
                                              'outcome', 'game_length', 'total_terrain_added',
                                              'total_bombs_placed']).to_csv('RL_performance.csv')