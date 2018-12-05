import os
import pandas as pd
from tqdm import tqdm
from utils import load_dict

def reward(bombs_placed, terrain_added):
    return bombs_placed * 10 + terrain_added * 10

for game in tqdm([x for x in os.listdir('data/') if '.' not in x]):

    cumulative_reward = 0

    rewards = []

    for frame_json in sorted([x for x in os.listdir('data/{}'.format(game)) if x.endswith('.pickle')], key=lambda x: int(x.replace('.pickle', ''))):
        json = load_dict('data/{}/{}'.format(game, frame_json))

        # extract the frame
        frame = json['game_properties']['frame']

        # bombs placed this frame
        bombs_placed = len([x for x in json['board_positions']['bombs_and_stage'] if x[2] == 1])

        # terrain added
        if frame == 1:
            amount_of_terrain = len(json['board_positions']['land'])

        terrain_added = len(json['board_positions']['land']) - amount_of_terrain
        amount_of_terrain = len(json['board_positions']['land'])

        # reward in this state
        added_reward = reward(bombs_placed, terrain_added)
        cumulative_reward += added_reward
        relative_reward = round(cumulative_reward / frame, 2)

        # save
        rewards.append([frame, added_reward, cumulative_reward, relative_reward])

    # write out per game
    pd.DataFrame(rewards, columns=['frame', 'added_reward', 'cumulative_reward', 'relative_reward'])\
        .to_csv('data/{}/rewards_total_{}_relative_{}.csv'.format(game, cumulative_reward, relative_reward))

