from game import Game, Player
from render_tool import RenderTool, MapScheme

import random

# choose the map
map = MapScheme().standard

# initialize the game
game = Game(map, verbose=True)
RT = RenderTool(game)

# name the players
player1 = Player(game, 'name_player1')
player2 = Player(game, 'name_player2')

# start the game
if game.start():
    RT.render_current_frame(save_media=True)
    game_status_dict = game.get_status_dict()

# play until the game is finished
while game_status_dict['game_properties']['outcome'] == 'ongoing':

    # select an action for the player
    move_player1 = random.choice([player1.Still, player1.Up, player1.Down, player1.Left, player1.Right, player1.Bomb])
    move_player1()

    move_player2 = random.choice([player2.Still, player2.Up, player2.Down, player2.Left, player2.Right, player2.Bomb])
    move_player2()

    # update the frame
    game.update_frame()
    game_status_dict = game.get_status_dict()

    # render the frame
    RT.render_current_frame(save_media=True)

