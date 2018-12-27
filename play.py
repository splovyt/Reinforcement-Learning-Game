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
    k = cv2.waitKey(0)
    if k == 27:
        break
    else:
        # UP
        if k == 122:
            player1.Up()
        # DOWN
        elif k == 115:
            player1.Down()
        # LEFT
        elif k == 113:
            player1.Left()
        # RIGHT
        elif k == 100:
            player1.Right()


        # BOMB
        elif k == 32:
            player1.Bomb()

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