import os
import shutil
from tqdm import tqdm
from utils import load_dict

MINIMUM_AMOUNT_OF_TURNS = 30
MINIMUM_AMOUNT_OF_BOMBS = 2

for game_id in tqdm([x for x in os.listdir('data') if '.' not in x]):
    # remove if we did not get enough turns
    if not os.path.exists('data/{}/{}.pickle'.format(game_id, MINIMUM_AMOUNT_OF_TURNS)):
        shutil.rmtree('data/{}'.format(game_id))

    else:
        all_json = [x for x in os.listdir('data/{}'.format(game_id)) if '.pickle' in x]

        # remove when there is no bomb placed
        bombs_placed = 0
        for json_file in sorted(all_json, key=lambda x: int(x.replace('.pickle', ''))):
            json = load_dict('data/{}/'.format(game_id) + json_file)
            stage_1_bombs = [x for x in json['board_positions']['bombs_and_stage'] if x[2] == 1]
            print(json['board_positions']['bombs_and_stage'])
            bombs_placed += len(stage_1_bombs)

        if bombs_placed < MINIMUM_AMOUNT_OF_BOMBS:
            shutil.rmtree('data/{}'.format(game_id))

os.system('python calculate_reward.py && python summarize_data.py')