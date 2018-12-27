from PIL import Image
import os
import cv2
import numpy as np

# STANDARD COLORS
class MapScheme:

    def __init__(self):

        # standard map scheme
        self.standard = {
            "name": 'standard',
            "background": {
                # void
                0: ([1, 1, 60, 255], 1),
                # land
                1: ([153, 255, 102, 255], 1),
                # block
                2: ([96, 63, 1, 255], 1),
            },
            "interactive": {
                # player 1
                -1: ([51, 153, 255, 0.5 * 255], 0.8),
                # player 2
                -2: ([212, 0, 255, 0.5 * 255], 0.8),
                # bomb
                1: ([0, 0, 0, 0.9 * 255], 0.5),
                # bomb stage 1
                2: ([244, 223, 66, 0.7 * 255], 0.7),
                # bomb stage 2
                3: ([255, 157, 0, 0.7 * 255], 0.7),
                # bomb stage 3
                4: ([255, 80, 0, 0.7 * 255], 0.7),
                # bomb explode
                5: ([255, 0, 0, 0.8 * 255], 0.7),
            }
        }

        # IBM map scheme
        self.IBM = {
            "name": 'IBM',
            "background": {
                # void
                0: ([51, 53, 56, 255], 1),
                # land
                1: ([255, 255, 255, 255], 1),
                # block
                2: ([7, 105, 186, 255], 1),
            },
            "interactive": {
                # player 1
                -1: ([51, 153, 255, 0.5 * 255], 0.8),
                # player 2
                -2: ([212, 0, 255, 0.5 * 255], 0.8),
                # bomb
                1: ([0, 0, 0, 0.9 * 255], 0.5),
                # bomb stage 1
                2: ([244, 223, 66, 0.7 * 255], 0.7),
                # bomb stage 2
                3: ([255, 157, 0, 0.7 * 255], 0.7),
                # bomb stage 3
                4: ([255, 80, 0, 0.7 * 255], 0.7),
                # bomb explode
                5: ([255, 0, 0, 0.8 * 255], 0.7),
            }
        }


class RenderTool:
    def __init__(self, game_obj):
        self.game = game_obj
        self.color_map = game_obj.map_scheme

        # get the size of the board
        self.y_tiles, self.x_tiles = self.game.board.shape

        # pixel size of a tile
        self.tile_size = 49

        # save frames
        self.image_path = 'data/{}/'.format(self.game.id)
        if not os.path.exists(self.image_path):
            os.makedirs(self.image_path)

    def insert_tile(self, image_pixels, row, column, new_tile, size=1.0):
        if size < 1.0:
            offset = int((new_tile.shape[0] - (size * new_tile.shape[0])) // 2)
            new_dim = int(new_tile.shape[0] - 2 * offset)
            new_tile = np.array(Image.fromarray(new_tile, 'RGBA').resize((new_dim, new_dim)), dtype='uint8')
            image_pixels[row * self.tile_size + offset:(row + 1) * self.tile_size - offset,
            column * self.tile_size + offset:(column + 1) * self.tile_size - offset, :] = new_tile
        else:
            image_pixels[row * self.tile_size:(row + 1) * self.tile_size,
            column * self.tile_size:(column + 1) * self.tile_size, :] = new_tile
        return image_pixels

    def render_video(self):
        init = True
        for image_path in [x for x in sorted([y for y in os.listdir(self.image_path) if '.png' in y],
                                             key=lambda x: int(x.split('.png')[0]))]:

            frame = cv2.imread(self.image_path + image_path)

            if init:
                height, width, channels = frame.shape
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
                out = cv2.VideoWriter(self.image_path + 'clip.mp4', fourcc, 1, (width, height))
                init = False

            # Write out frame to video
            out.write(frame)  #

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()

    def render_current_frame(self, save_media=True):
        # create blank image
        img = Image.new("RGBA", (self.x_tiles * self.tile_size, self.y_tiles * self.tile_size), "white")

        # image to pixels
        img_bg_pixels = np.array(img, dtype='uint8')

        # color pixels using the color map
        ## background ##
        for r in range(self.game.board.shape[0]):
            for c in range(self.game.board.shape[1]):
                # create tile
                rgb_color = self.color_map["background"][self.game.board[r, c]][0]
                size = self.color_map["background"][self.game.board[r, c]][1]
                new_tile = np.full((self.tile_size, self.tile_size, 4), rgb_color)
                img_bg_pixels = self.insert_tile(img_bg_pixels, r, c, new_tile, size)

        # pixels to image
        img_bg = Image.fromarray(img_bg_pixels, 'RGBA')

        ## foreground ##
        for obj_id in self.color_map["interactive"].keys():
            img_fg_pixels = np.full(np.array(img_bg).shape, [255, 255, 255, 0], dtype='uint8')
            for r in range(self.game.board.shape[0]):
                for c in range(self.game.board.shape[1]):
                    try:
                        obj_set = self.game.movable_objects[r][c].copy()
                        if obj_id in obj_set:
                            # create tile
                            rgb_color = self.color_map["interactive"][obj_id][0]
                            size = self.color_map["interactive"][obj_id][1]
                            new_tile = np.full((self.tile_size, self.tile_size, 4), rgb_color, dtype='uint8')
                            img_fg_pixels = self.insert_tile(img_fg_pixels.copy(), r, c, new_tile, size)
                    except Exception as exc:
                        print(exc)

            # pixels to image
            img_fg_temp = Image.fromarray(img_fg_pixels, 'RGBA')
            try:
                img_fg = Image.alpha_composite(img_fg, img_fg_temp)
            except:
                img_fg = img_fg_temp

        final_img = Image.alpha_composite(img_bg, img_fg)
        if save_media:
            final_img.save(self.image_path + str(self.game.frame) + '.png', 'PNG')

            if self.game.ended:
                self.render_video()

        return final_img.convert('RGB')
