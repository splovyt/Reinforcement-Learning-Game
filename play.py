from game import Game, Player
from render_tool import RenderTool, MapScheme

import cv2
import traceback
import random

# choose the map
map = MapScheme().standard

# initialize the game
game = Game(map)
RT = RenderTool(game)

player1 = Player(game, 'player1')
player2 = Player(game, 'player2')

if game.start():
    img = RT.render_current_frame(save_media=True)

img = cv2.imread(RT.image_path + '{}.png'.format(game.frame))
cv2.namedWindow(game.id)
cv2.imshow(game.id, img)

while 1:
    # the pressed key
    k = cv2.waitKey(0)
    if k == 27: # ESCAPE KEY
        print('Escape key pressed. Closing game..')
        break
    else:
        # UP (W key)
        if k == 119: # change to 122 (Z) on azerty keyboards
            player1.Up()
        # DOWN (S Key)
        elif k == 115:
            player1.Down()
        # LEFT (A key)
        elif k == 97: # change to 113 (Q) on azerty keyboards
            player1.Left()
        # RIGHT (D key)
        elif k == 100:
            player1.Right()


        # BOMB (SPACE BAR)
        elif k == 32:
            player1.Bomb()

        # If any other key is pressed (by accident), we stand still
        else:
            print("Not recognized key {} pressed. Standing still for this action..".format(k))
            player1.Still()

        # choose a random action for the other player (minus bomb to make sure that player2 does not blow itself up)
        move_func = random.choice([player2.Still, player2.Up, player2.Down, player2.Left, player2.Right])
        move_func()

        try:
            game.update_frame()
            RT.render_current_frame(save_media=True)
            img = cv2.imread(RT.image_path + '{}.png'.format(game.frame))
            cv2.imshow(game.id, img)
        except Exception as exc:
            print(exc)
            print(traceback.format_exc())
            break