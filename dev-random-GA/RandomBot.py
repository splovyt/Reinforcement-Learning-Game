# source imports
from game import Game, Player
from render_tool import RenderTool, MapScheme

# additional library imports
import random
import pandas as pd

# dev-RL-solution imports
from utils import save_dict
from tqdm import tqdm



for _ in tqdm(range(5000)):
    # choose the map
    #map = MapScheme().IBM
    map = MapScheme().standard

    # initialize the game
    game = Game(map, verbose=False)
    RT = RenderTool(game)

    # name the players
    player1 = Player(game, 'Sonic')
    player2 = Player(game, 'Knuckles')

    # start the game
    if game.start():
        RT.render_current_frame(save_media = False)
        game_status_dict = game.get_status_dict()
        save_dict('data/{}/{}.pickle'.format(game.id, game.frame), game_status_dict)

    while game_status_dict['game_properties']['outcome'] == 'ongoing':

        # select an action for the player
        # move_player1 = random.choice([player1.Still, player1.Up, player1.Down, player1.Left, player1.Right, player1.Bomb])
        move_player1 = player1.Still
        move_player1()

        if game.frame == 1:
            move_player2 = random.choice([player2.Up, player2.Left])
        elif game.frame == 2:
            move_player2 = player2.Bomb
        else:
            #move_player2 = random.choice([player2.Still, player2.Up, player2.Down, player2.Left, player2.Right, player2.Bomb])
            move_player2 = random.choice([player2.Up, player2.Down, player2.Left, player2.Right, player2.Bomb])
            #move_player2 = player2.Still
        move_player2()

        # update the frame
        game.update_frame()
        game_status_dict = game.get_status_dict()
        save_dict('data/{}/{}.pickle'.format(game.id, game.frame), game_status_dict)

        # render the frame
        RT.render_current_frame(save_media = False)

    # the outcome
    if game.verbose:
        print('outcome: {}'.format(game_status_dict['game_properties']['outcome']))

    # save history
    moves_history = pd.DataFrame(columns=['player_1', 'player_2'])
    moves_history['player_1'] = game.players[0].history
    moves_history['player_2'] = game.players[1].history
    moves_history.to_csv('data/{}/moves.csv'.format(game.id))
