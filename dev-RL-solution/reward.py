'''This file contains utility functions related to the reward system of the DQN algorithm.'''

def reward_function(bombs_placed, terrain_added, win_lose_ongoing):
    # if the game ended
    if win_lose_ongoing == 'win':
        return 0
        #return 100
    elif win_lose_ongoing == 'lose':  # draw is a loss too
        return -50

    # if nothing happened, place a penalty for duration of the game
    elif bombs_placed == 0 and terrain_added == 0:
        return -1

    # calculate the reward
    else:
        return bombs_placed * 10 + terrain_added * 50


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