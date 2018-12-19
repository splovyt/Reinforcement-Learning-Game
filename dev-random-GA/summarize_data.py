import os, glob
import pandas as pd
from tqdm import tqdm

summary_df = []
for game in tqdm(os.listdir('data/')):
    try:
        rewards_df = pd.read_csv(glob.glob('data/{}/reward*relative*.csv'.format(game))[0])


        relative_reward = rewards_df['relative_reward'].tolist()[-1]
        cumulative_reward = rewards_df['cumulative_reward'].tolist()[-1]
        frames = rewards_df['frame'].tolist()[-1]

        summary_df.append([game, frames, cumulative_reward, relative_reward])

    except IndexError:
        pass
        #print('calculate the rewards first')

summary_df = pd.DataFrame(summary_df, columns=['game', 'frames', 'cumulative_reward', 'relative_reward'])\
    .sort_values('relative_reward', ascending=False)
summary_df.to_csv('data_summary.csv')
print(summary_df)

