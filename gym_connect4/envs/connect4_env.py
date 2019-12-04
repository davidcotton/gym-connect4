import copy
from typing import List, Optional, Set, Tuple

import gym
from gym import spaces
import numpy as np


# default game config, can be overridden in `env_config`
BOARD_HEIGHT = 6
BOARD_WIDTH = 7
WIN_LENGTH = 4
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
REWARD_DRAW = 0.0
REWARD_STEP = 0.0


class Connect4Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None) -> None:
        super().__init__()
        self.game = Connect4(env_config)
        self.action_space = spaces.Discrete(self.game.board_width)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.game.board_height, self.game.board_width), dtype=np.uint8)
        self.board1 = np.zeros((self.game.board_height, self.game.board_width), dtype=np.uint8)
        self.board2 = np.zeros((self.game.board_height, self.game.board_width), dtype=np.uint8)

    def reset(self):
        self.game = Connect4()
        self.board1 = np.zeros((self.game.board_height, self.game.board_width), dtype=np.uint8)
        self.board2 = np.zeros((self.game.board_height, self.game.board_width), dtype=np.uint8)
        return self.get_state(0), self.get_state(1)

    def step(self, column):
        """Make a game action.

        Throws a ValueError if trying to drop into a full column.

        :param column: The column index to drop in ([0-7]).
        :return: A tuple containing the next state, reward, if the game ended and an empty info dict.
        """
        if not self.game.is_valid_move(column):
            raise ValueError('Invalid action, column %s is full' % column)
        self.game.move(column)

        self.board1[self.game.lowest_row[column] - 1][column] = self.game.player + 1
        self.board2[self.game.lowest_row[column] - 1][column] = (self.game.player ^ 1) + 1

        state = self.get_state(0), self.get_state(1)
        reward = self.game.get_reward()
        game_over = self.game.is_game_over()

        return state, reward, game_over, {}

    def valid_actions(self) -> Set[int]:
        """Fetch a set of valid moves available.

        :return: A set of valid moves.
        """
        valid_moves = set(self.game.get_moves())
        return valid_moves

    def get_state(self, player=None) -> np.ndarray:
        if player == 1:
            board = self.board2.copy()
        else:
            board = self.board1.copy()
        state = np.flip(board, axis=0)
        return state

    def render(self, mode='human') -> None:
        print('  1 2 3 4 5 6 7')
        print(' ---------------')
        print(self.get_state())
        print(' ---------------')
        print('  1 2 3 4 5 6 7')
        # print()
        # if self.game.is_game_over():
        #     print('##########')
        #     print('Winner:', self.game.player + 1)
        #     print('##########')

    def winner(self) -> int:
        """Fetch the winner of a completed game.

        :return: 0 or 1 if that player is winner, -1 for draw.
        """
        assert self.game.is_game_over()
        if self.game.is_draw():
            return -1
        return 0 if self.game.is_winner(0) else 1

    def time(self):
        return np.count_nonzero(self.board1)

    @property
    def reward_win(self):
        return self.game.reward_win

    @property
    def reward_lose(self):
        return self.game.reward_lose

    @property
    def reward_draw(self):
        return self.game.reward_draw


class Connect4:
    def __init__(self, env_config=None) -> None:
        super().__init__()
        self.env_config = dict({
            'board_height': BOARD_HEIGHT,
            'board_width': BOARD_WIDTH,
            'win_length': WIN_LENGTH,
            'reward_win': REWARD_WIN,
            'reward_draw': REWARD_DRAW,
            'reward_lose': REWARD_LOSE,
            'reward_step': REWARD_STEP,
        }, **env_config or {})

        # self.board = np.zeros((self.board_height, self.board_width), dtype=np.uint8)
        self.bit_board = [0, 0]  # bit-board for each player
        self.dirs = [1, (self.board_height + 1), (self.board_height + 1) - 1, (self.board_height + 1) + 1]  # this is used for bitwise operations
        self.heights = [(self.board_height + 1) * i for i in range(self.board_width)]  # top empty row for each column
        self.lowest_row = [0] * self.board_width  # number of stones in each row
        self.top_row = [(x * (self.board_height + 1)) - 1 for x in range(1, self.board_width + 1)]  # top row of the board (this will never change)
        self.player = 1

    def clone(self):
        clone = Connect4()
        # clone.board = copy.deepcopy(self.board)
        clone.bit_board = copy.deepcopy(self.bit_board)
        clone.heights = copy.deepcopy(self.heights)
        clone.lowest_row = copy.deepcopy(self.lowest_row)
        clone.top_row = copy.deepcopy(self.top_row)
        clone.player = self.player
        return clone

    def move(self, column) -> None:
        m2 = 1 << self.heights[column]  # position entry on bit-board
        self.heights[column] += 1  # update top empty row for column
        self.player ^= 1
        self.bit_board[self.player] ^= m2  # XOR operation to insert stone in player's bit-board
        # self.board[self.lowest_row[column]][column] = self.player + 1  # update entry in matrix (only for printing)
        self.lowest_row[column] += 1  # update number of stones in column

    def get_reward(self, player=None) -> float:
        if player is None:
            player = self.player

        if self.is_winner(player):
            return self.reward_win
        elif self.is_winner(player ^ 1):
            return self.reward_lose
        elif self.is_draw():
            return self.reward_draw
        else:
            return self.reward_step

    def is_winner(self, player: int = None) -> bool:
        """Evaluate board, find out if a player has won.

        :param player: The player to check.
        :return: True if the player has won, otherwise False.
        """
        if player is None:
            player = self.player

        for d in self.dirs:
            bb = self.bit_board[player]
            for i in range(1, self.win_length):
                bb &= self.bit_board[player] >> (i * d)
            if bb != 0:
                return True
        return False

    def is_draw(self) -> bool:
        """Is the game a draw?

        :return: True if the game is drawn, else False.
        """
        return not self.get_moves() and not self.is_winner(self.player) and not self.is_winner(self.player ^ 1)

    def is_game_over(self) -> bool:
        """Is the game over?

        :return: True if the game is over, else False.
        """
        return self.is_winner(self.player) or self.is_winner(self.player ^ 1) or not self.get_moves()

    # returns list of available moves
    def get_moves(self) -> List[int]:
        if self.is_winner(self.player) or self.is_winner(self.player ^ 1):
            return []  # if terminal state, return empty list

        list_moves = []
        for i in range(self.board_width):
            if self.lowest_row[i] < self.board_height:
                list_moves.append(i)
        return list_moves

    # def valid_actions(self) -> Set[int]:
    #     """Fetch a set of valid moves available.
    #
    #     :return: A set of valid moves.
    #     """
    #     valid_moves = set()
    #     for column in range(self.board.shape[1]):
    #         if self.is_valid_move(column):
    #             valid_moves.add(column)
    #     return valid_moves

    def is_valid_move(self, column: int) -> bool:
        """Check if column is full.

        :param column: The column to check
        :return: True if it is a valid move, else False.
        """
        return self.heights[column] != self.top_row[column]

    @property
    def board_height(self) -> int:
        return self.env_config['board_height']

    @property
    def board_width(self) -> int:
        return self.env_config['board_width']

    @property
    def win_length(self) -> int:
        return self.env_config['win_length']

    @property
    def reward_win(self) -> float:
        return self.env_config['reward_win']

    @property
    def reward_draw(self) -> float:
        return self.env_config['reward_draw']

    @property
    def reward_lose(self) -> float:
        return self.env_config['reward_lose']

    @property
    def reward_step(self) -> float:
        return self.env_config['reward_step']
