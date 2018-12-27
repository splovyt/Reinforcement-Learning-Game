import numpy as np
from uuid import uuid4 as random_id_generator
from functools import wraps

class Game:

    def __init__(self, map_scheme, verbose=False):
        '''Initialize the game.'''
        self.verbose = verbose

        ## initialize the map scheme ##
        self.map_scheme = map_scheme

        ## assign a random ID to the game ##
        self.id = str(random_id_generator())

        ## start frame ##
        self.frame = 0
        self.ended = False

        map_type = self.map_scheme["name"]
        if map_type == 'standard':
            ## initialize board with the specified dimensions ##
            self.board = np.zeros((7, 11))
            # movable objects is a dictionary of lists with the following format: {}[row][column] = [objects]
            self.movable_objects = dict()
            self._bomb_dict = dict()
            for row in range(self.board.shape[0]):
                self.movable_objects[row] = dict()
                self._bomb_dict[row] = dict()
                for column in range(self.board.shape[1]):
                    self._bomb_dict[row][column] = 0
                    self.movable_objects[row][column] = set()

            ## add starting floor to the map ##
            # upper left
            self.board[0, 0], self.board[0, 1], self.board[1, 0] = 1, 1, 1
            # upper right
            self.board[0, -1], self.board[0, -2], self.board[1, -1] = 1, 1, 1
            # bottom left
            self.board[-1, 0], self.board[-2, 0], self.board[-1, 1] = 1, 1, 1
            # bottom right
            self.board[-1, -1], self.board[-2, -1], self.board[-1, -2] = 1, 1, 1

            ## add blocks ##
            self.board[1, 1], self.board[2, 2], self.board[2, 1] = 2, 2, 2
            self.board[-2, 1], self.board[-3, 1], self.board[-3, 2] = 2, 2, 2
            self.board[-2, -2], self.board[-3, -3], self.board[-3, -2] = 2, 2, 2
            self.board[1, -2], self.board[2, -3], self.board[2, -2] = 2, 2, 2

            self.board[0, 3], self.board[0, -4] = 2, 2
            self.board[-1, 3], self.board[-1, -4] = 2, 2

            self.board[2, 5], self.board[3, 5], self.board[4, 5], self.board[3, 4], self.board[3, -5] = 2, 2, 2, 2, 2

        elif map_type == 'IBM':
            ## initialize board with the specified dimensions ##
            self.board = np.zeros((15, 22))
            # movable objects is a dictionary of lists with the following format: {}[row][column] = [objects]
            self.movable_objects = dict()
            self._bomb_dict = dict()
            for row in range(self.board.shape[0]):
                self.movable_objects[row] = dict()
                self._bomb_dict[row] = dict()
                for column in range(self.board.shape[1]):
                    self._bomb_dict[row][column] = 0
                    self.movable_objects[row][column] = set()

            ## add starting floor to the map ##
            # upper left
            self.board[0, 0], self.board[0, 1], self.board[1, 0] = 1, 1, 1
            # upper right
            self.board[0, -1], self.board[0, -2], self.board[1, -1] = 1, 1, 1
            # bottom left
            self.board[-1, 0], self.board[-2, 0], self.board[-1, 1] = 1, 1, 1
            # bottom right
            self.board[-1, -1], self.board[-2, -1], self.board[-1, -2] = 1, 1, 1

            ## add blocks ##
            # I
            self.board[3, 2], self.board[3, 3], self.board[5, 2], self.board[5, 3] = 2, 2, 2, 2
            self.board[7, 2], self.board[7, 3], self.board[9, 2], self.board[9, 3] = 2, 2, 2, 2
            self.board[11, 2], self.board[11, 3] = 2, 2

            # B
            self.board[3, 6], self.board[3, 7], self.board[3, 9] = 2, 2, 2
            self.board[5, 7], self.board[5, 10] = 2, 2
            self.board[7, 7], self.board[7, 9] = 2, 2
            self.board[9, 7], self.board[9, 10] = 2, 2
            self.board[11, 6], self.board[11, 7], self.board[11, 9] = 2, 2, 2

            # M
            self.board[3, 13], self.board[3, 15], self.board[3, 17], self.board[3, 19] = 2, 2, 2, 2
            self.board[5, 13], self.board[5, 16], self.board[5, 19] = 2, 2, 2
            self.board[7, 13], self.board[7, 16], self.board[7, 19] = 2, 2, 2
            self.board[9, 13], self.board[9, 19] = 2, 2
            self.board[11, 13], self.board[11, 19] = 2, 2

        ## define player starting positions (this also defines how many players are available) ##
        self.player_name_to_object = dict()
        # 2 players
        self.player_slots = [(0, 0), (self.board.shape[0] - 1, self.board.shape[1] - 1)]
        self.players = []
        self.player_action_queue = dict()

    def _add_player(self, player_instance):
        assert player_instance.name not in [p.name for p in
                                            self.players], 'There is already a player with the name {}'.format(
            player_instance.name)
        assert len(self.players) < len(self.player_slots), 'All players are already loaded in.'

        # append player to player list
        self.players.append(player_instance)

        # assign id to the player
        player_instance.id = -(len(self.players))

        # update dictionaries
        self.player_name_to_object[player_instance.name] = player_instance

        # assign starting position to the player
        player_instance.position = self.player_slots[len(self.players) - 1]
        self.movable_objects[player_instance.position[0]][player_instance.position[1]].update([player_instance.id])

        # initialize actions for the player
        self.player_action_queue[player_instance.name] = None

    def _possible_move(self, next_location_r, next_location_c):
        # if we are at the edge of the map
        if next_location_r not in range(self.board.shape[0]) or next_location_c not in range(self.board.shape[1]):
            return False
        # if the next location is empty
        if self.board[next_location_r, next_location_c] == 0:
            return False
        # if the next location is a block
        if self.board[next_location_r, next_location_c] == 2:
            return False
        # if the next location is a bomb
        if 1 in self.movable_objects[next_location_r][next_location_c]:
            return False
        # if no blocks, return True
        return True

    def start(self):
        assert len(self.players) == len(self.player_slots), 'Not all players are loaded in yet.'
        assert self.frame == 0, 'The game is already ongoing.'
        self.frame = 1
        if self.verbose:
            print('The game has been started! (id: {})'.format(self.id))
        return True

    def _move_player(self, new_r, new_c, player):
        # remove player from old position
        self.movable_objects[player.position[0]][player.position[1]].remove(player.id)
        # update player position to new position
        self.movable_objects[new_r][new_c].update([player.id])
        player.position = (new_r, new_c)

    def _bomb(self, pos_r, pos_c, blast_range=2):
        self.movable_objects[pos_r][pos_c].update([1])

        # make new bomb tracker
        self._bomb_dict[pos_r][pos_c] = 1

        # add blast range
        advance_r_plus = True
        advance_r_min = True
        advance_c_plus = True
        advance_c_min = True
        for b in range(blast_range + 1):
            # DOWN
            if pos_r + b in range(self.board.shape[0]):
                if self.board[pos_r + b, pos_c] != 2 and advance_r_plus:
                    # if it's not blocked and we didnt encounter one before
                    self.movable_objects[pos_r + b][pos_c].update([2])
                else:
                    # if it's blocked, don't advance anymore
                    advance_r_plus = False
            # UP
            if pos_r - b in range(self.board.shape[0]):
                if self.board[pos_r - b, pos_c] != 2 and advance_r_min:
                    # if it's not blocked and we didnt encounter one before
                    self.movable_objects[pos_r - b][pos_c].update([2])
                else:
                    # if it's blocked, don't advance anymore
                    advance_r_min = False
            # RIGHT
            if pos_c + b in range(self.board.shape[1]):
                if self.board[pos_r, pos_c + b] != 2 and advance_c_plus:
                    # if it's not blocked and we didnt encounter one before
                    self.movable_objects[pos_r][pos_c + b].update([2])
                else:
                    # if it's blocked, don't advance anymore
                    advance_c_plus = False
            # LEFT
            if pos_c - b in range(self.board.shape[1]):
                if self.board[pos_r, pos_c - b] != 2 and advance_c_min:
                    # if it's not blocked and we didnt encounter one before
                    self.movable_objects[pos_r][pos_c - b].update([2])
                else:
                    # if it's blocked, don't advance anymore
                    advance_c_min = False

    def _update_blast(self):
        for r in self.movable_objects.keys():
            for c, value_set in self.movable_objects.copy()[r].items():
                # advance / remove bombs
                if self._bomb_dict[r][c] in [1, 2, 3]:
                    # advance bomb one level
                    self._bomb_dict[r][c] += 1
                if self._bomb_dict[r][c] == 4:
                    # bomb exploded; remove it
                    self._bomb_dict[r][c] = 0
                    self.movable_objects[r][c].remove(1)

                # turn completed blasts into new ground
                if 5 in self.movable_objects[r][c]:
                    self.board[r, c] = 1
                    self.movable_objects[r][c].remove(5)

                # update blast
                if 4 in value_set:
                    self.movable_objects[r][c].remove(4)
                    self.movable_objects[r][c].update([5])
                if 3 in value_set:
                    self.movable_objects[r][c].remove(3)
                    self.movable_objects[r][c].update([4])
                if 2 in value_set:
                    self.movable_objects[r][c].remove(2)
                    self.movable_objects[r][c].update([3])

    def check_players_status(self):
        player_status = dict()
        for player in self.players:
            player_r, player_c = player.position
            # if bomb detonation affects the player position
            if 5 in self.movable_objects[player_r][player_c]:
                player.alive = False
            player_status[player] = player.alive
        return player_status

    def check_game_status(self):
        # if the amount of active players is less or equal than 1
        if list(self.check_players_status().values()).count(True) <= 1:
            self.ended = True
            if self.verbose:
                print('GAME OVER')
        return

    def update_frame(self):
        if self.ended:
            print('The game has already ended.')
            return False
        assert self.frame > 0, 'Start the game first'
        assert None not in [value for key, value in
                            self.player_action_queue.items()], 'Not all player actions have been defined yet'


        # update frame
        self.frame += 1

        # update blast radius
        self._update_blast()

        # update the player positions on the board
        for player_name, action in self.player_action_queue.items():
            # get player instance
            player = self.player_name_to_object[player_name]

            if isinstance(action, tuple):
                # move player
                new_position_r, new_position_c = action
                self._move_player(new_position_r, new_position_c, player)
            elif isinstance(action, str):
                if action == 'bomb':
                    # drop bomb
                    self._bomb(player.position[0], player.position[1])
            # empty action queue
            self.player_action_queue[player_name] = None

        # check if game ends (Player 1 win, Player 2 win, Draw)
        self.check_game_status()
        return True

    def get_status_dict(self):
        d = {}

        ## GAME PROPERTIES ##
        d['game_properties'] = {}

        # add game id
        d['game_properties']['id'] = self.id

        # add board dimensions
        d['game_properties']['board_dimensions'] = self.board.shape

        # add game status
        if self.ended:
            draw = True
            for player in self.players:
                if player.alive:
                    d['game_properties']['outcome'] = 'player_{}'.format(-player.id)
                    draw = False
            if draw:
                d['game_properties']['outcome'] = 'draw'
        else:
            d['game_properties']['outcome'] = 'ongoing'

        # add frame
        d['game_properties']['frame'] = self.frame

        ## BOARD POSITIONS ##
        d['board_positions'] = {}

        # add player positions
        d['board_positions']['players'] = {}
        for player in self.players:
            d['board_positions']['players']['player_{}'.format(-player.id)] = player.position

        # add void positions
        void_pos = np.where(self.board == 0)
        d['board_positions']['void'] = [(void_pos[0][idx], void_pos[1][idx]) for idx in range(len(void_pos[0]))]

        # add land positions
        land_pos = np.where(self.board == 1)
        d['board_positions']['land'] = [(land_pos[0][idx], land_pos[1][idx]) for idx in range(len(land_pos[0]))]

        # add block positions
        block_pos = np.where(self.board == 2)
        d['board_positions']['block'] = [(block_pos[0][idx], block_pos[1][idx]) for idx in range(len(block_pos[0]))]

        # add bomb positions and set stage
        bomb_pos = []
        for r in self._bomb_dict.keys():
            for c in self._bomb_dict[r].keys():
                stage = self._bomb_dict[r][c]
                # if there is a bomb, save the stage
                if stage >= 1:
                    bomb_pos.append((r, c, stage))
        d['board_positions']['bombs_and_stage'] = bomb_pos

        # add blast radius and stage
        for blast_stage in [2, 3, 4, 5]:
            # create field
            field = 'blast_radius_{}'.format(blast_stage - 1)
            d['board_positions'][field] = []

            # check where to find this blast stage
            for r in self.movable_objects.keys():
                for c in self.movable_objects[r].keys():
                    # if this particular blast stage is found
                    if blast_stage in self.movable_objects[r][c]:
                        d['board_positions'][field].append((r, c))

        return d


def validate(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        player = args[0]
        assert player.game.frame > 0, 'You must start the game first! Use game.start().'
        func(*args, **kwargs)
    return wrapper


class Player:
    def __init__(self, game_instance, player_name):
        self.game = game_instance
        self.name = player_name
        self.id = None
        self.alive = True
        self.position = (None, None)

        # add player to the game
        self.game._add_player(self)

        # move history
        self.history = []

    @validate
    def Up(self):
        current_location_r, current_location_c = self.position
        next_location_r, next_location_c = current_location_r - 1, current_location_c

        if not self.game._possible_move(next_location_r, next_location_c):
            # The next step is not possible, stay in the current location
            next_location_c = current_location_c
            next_location_r = current_location_r

        # add next to queue
        self.game.player_action_queue[self.name] = (next_location_r, next_location_c)

        # save move
        self.history.append('up')

    @validate
    def Down(self):
        current_location_r, current_location_c = self.position
        next_location_r, next_location_c = current_location_r + 1, current_location_c

        if not self.game._possible_move(next_location_r, next_location_c):
            # The next step is not possible, stay in the current location
            next_location_c = current_location_c
            next_location_r = current_location_r

        # add next to queue
        self.game.player_action_queue[self.name] = (next_location_r, next_location_c)

        # save move
        self.history.append('down')

    @validate
    def Left(self):
        current_location_r, current_location_c = self.position
        next_location_r, next_location_c = current_location_r, current_location_c - 1

        if not self.game._possible_move(next_location_r, next_location_c):
            # The next step is not possible, stay in the current location
            next_location_c = current_location_c
            next_location_r = current_location_r

        # add next to queue
        self.game.player_action_queue[self.name] = (next_location_r, next_location_c)

        # save move
        self.history.append('left')

    @validate
    def Right(self):
        current_location_r, current_location_c = self.position
        next_location_r, next_location_c = current_location_r, current_location_c + 1

        if not self.game._possible_move(next_location_r, next_location_c):
            # The next step is not possible, stay in the current location
            next_location_c = current_location_c
            next_location_r = current_location_r

        # add next to queue
        self.game.player_action_queue[self.name] = (next_location_r, next_location_c)

        # save move
        self.history.append('right')

    @validate
    def Still(self):
        current_location_r, current_location_c = self.position
        next_location_r, next_location_c = current_location_r, current_location_c

        # add next to queue
        self.game.player_action_queue[self.name] = (next_location_r, next_location_c)

        # save move
        self.history.append('still')

    @validate
    def Bomb(self):
        # add next to queue
        self.game.player_action_queue[self.name] = 'bomb'

        # save move
        self.history.append('bomb')