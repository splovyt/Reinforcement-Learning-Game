# source imports
from game import Game, Player
from render_tool import RenderTool, MapScheme

# DQN imports
from settings import * # the config file
from DQN import DQN_RL # the agent
from DQN_utils import reshape_state
from reward import calculate_reward, calculate_bombs_placed, calculate_terrain_added

# additional library imports
import math
import random
import pandas as pd
import numpy as np

# dev imports
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
                                                   #'Still': player1.Still,
                                                   'Bomb': player1.Bomb})
    agent_player2.initialize_actions(actions_dict={'Up': player2.Up,
                                                   'Down': player2.Down,
                                                   'Left': player2.Left,
                                                   'Right': player2.Right,
                                                   #'Still': player2.Still,
                                                   'Bomb': player2.Bomb})

    # calculate the properties of the input images
    pixel_height_tile = math.ceil(math.sqrt(PIXELS_PER_TILE))
    pixel_width_tile = pixel_height_tile
    input_image_height = pixel_height_tile * game.board.shape[0]
    input_image_width = pixel_width_tile * game.board.shape[1]

    # build the CNN
    agent_player1.buildCNN(game, input_image_height, input_image_width)
    agent_player2.buildCNN(game, input_image_height, input_image_width)

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

    # require at least some terrain addition
    if len(agent_player1.q_table) > BATCH_SIZE:
        agent_player1.reinforce()
        agent_player2.reinforce()

    # track performance
    RL_performance.append([game.id, round(agent_player1.epsilon,2), cumulative_reward_player1, cumulative_reward_player2,
                           game_status_dict['game_properties']['outcome'], len(game.players[0].history),
                           total_terrain_added, total_bombs_placed])

    if e > 0 and e % 10 == 0:
        pd.DataFrame(RL_performance, columns=['game_id','epsilon', 'cum_reward_p1', 'cum_reward_p2',
                                              'outcome', 'game_length', 'total_terrain_added',
                                              'total_bombs_placed']).to_csv('RL_performance.csv')