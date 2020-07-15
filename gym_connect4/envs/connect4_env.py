import copy
from typing import List

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

# making these constants for clarity
WINNER_NONE = -2
WINNER_DRAW = -1
WINNER_PLAYER0 = 0
WINNER_PLAYER1 = 1


class Connect4Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None) -> None:
        super().__init__()
        self.game = Connect4(env_config)
        self.action_space = spaces.Discrete(self.game.board_width + 1)
        self.observation_space = spaces.Dict({
            'action_mask': spaces.Box(low=0, high=1, shape=(self.game.board_width + 1,), dtype=np.uint8),
            'board': spaces.Box(low=0, high=2, shape=(self.game.board_height, self.game.board_width), dtype=np.uint8),
            # 'current_player': spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            # 'player_id': spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            # 'winner': spaces.Box(low=-2, high=1, shape=(1,), dtype=np.int8),
        })
        # maintain a copy of each player's observations
        # each board is player invariant, has the player as `1` and the opponent as `2`
        self.boards: List[np.array] = []

    def reset(self):
        self.game = Connect4()
        self.boards = [np.zeros((self.game.board_height, self.game.board_width), dtype=np.uint8) for _ in range(2)]
        obs_dict = {
            i: {
                'action_mask': self._get_action_mask(i),
                'board': self._get_state(i),
                # 'current_player': np.array([0]),  # player0 is always first
                # 'player_id': np.array([i]),
                # 'winner': WINNER_NONE
            } for i in range(2)
        }
        return obs_dict

    def step(self, actions: List[int]):
        """Make a game action.

        Throws a ValueError if trying to drop into a full column.

        :param actions: A list ints encoding player: action.
        :return: A tuple containing the next obs, rewards, if the game ended and an empty info dict.
        """

        player = self.game.player ^ 1  # game.player is incremented in game.move(), so use flipped value internally
        # next_player = self.game.player
        column = actions[player]

        try:
            assert self.game.is_valid_move(column)
        except Exception as e:
            print('Invalid action, column %s is full' % column)
            print(self._get_state(player))
            raise e

        self.game.move(column)
        self.boards[0][self.game.column_counts[column] - 1][column] = self.game.player + 1
        self.boards[1][self.game.column_counts[column] - 1][column] = (self.game.player ^ 1) + 1

        game_over = self.game.is_game_over()
        # winner = self._get_winner() if game_over else WINNER_NONE
        obs = {
            i: {
                'action_mask': self._get_action_mask(i),
                'board': self._get_state(i),
                # 'current_player': np.array([next_player]),
                # 'player_id': np.array([i]),
                # 'winner': winner,
            } for i in range(2)
        }
        rewards = {i: self.game.get_reward(i) for i in range(2)}

        return obs, rewards, game_over, {}

    def render(self, mode='human') -> None:
        print('  1 2 3 4 5 6 7')
        print(' ---------------')
        print(self._get_state())
        print(' ---------------')
        print('  1 2 3 4 5 6 7')

    def _get_state(self, player=None) -> np.ndarray:
        if player == 0 or player is None:
            board = self.boards[0]
        elif player == 1:
            board = self.boards[1]
        else:
            raise ValueError('Invalid player ID %s' % player)
        return np.flip(board, axis=0).copy()

    def _get_action_mask(self, player) -> np.ndarray:
        if player == self.game.player ^ 1:
            mask = np.array(self.game.get_action_mask() + [0], dtype=np.uint8)
        else:
            mask = np.zeros((8,), dtype=np.uint8)
            mask[-1] = 1
        return mask

    def _get_winner(self) -> int:
        if self.game.is_draw():
            return WINNER_DRAW
        return WINNER_PLAYER0 if self.game.is_winner(0) else WINNER_PLAYER1

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
    def __init__(self, env_config=None, game_state=None) -> None:
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

        self.player = 1  # current player (in {0, 1})
        self.bitboard = [0, 0]  # bitboard for each player
        # the four different win condition directions to bitshift over:
        #   - (vertical, horizontal, diagonal-descending, diagonal-ascending)
        self.win_conditions = [1, (self.board_height + 1), (self.board_height + 1) - 1, (self.board_height + 1) + 1]
        # index of the bottom empty space for each column
        self.empty_indexes = [(self.board_height + 1) * i for i in range(self.board_width)]
        # number of discs in each column
        self.column_counts = [0] * self.board_width
        # to check for valid moves it is convenient to build an index of the top row of the board to compare against
        self.top_row = [(x * (self.board_height + 1)) - 1 for x in range(1, self.board_width + 1)]

        # cache winner checks (each turn) as a little expensive
        self.is_winner_cache = [None, None]

        if game_state is not None:  # reconstitute from game state
            board = np.flip(game_state['board'].reshape(self.board_height, self.board_width), axis=0).astype(np.uint8)
            self.player = game_state['player']
            for y, row in enumerate(board):
                num_updated = 0
                for column, value in enumerate(row):
                    if value in {1, 2}:
                        player = value - 1
                        m2 = 1 << self.empty_indexes[column]
                        self.empty_indexes[column] += 1
                        self.bitboard[player] ^= m2
                        self.column_counts[column] += 1
                        num_updated += 1
                if num_updated == 0:
                    break

    def clone(self):
        clone = Connect4()
        clone.bitboard = copy.deepcopy(self.bitboard)
        clone.empty_indexes = copy.deepcopy(self.empty_indexes)
        clone.column_counts = copy.deepcopy(self.column_counts)
        clone.top_row = copy.deepcopy(self.top_row)
        clone.player = self.player
        return clone

    def move(self, column) -> None:
        self.is_winner_cache = [None, None]
        m2 = 1 << self.empty_indexes[column]  # position entry on bitboard
        self.empty_indexes[column] += 1  # update top empty row for column
        self.player ^= 1
        self.bitboard[self.player] ^= m2  # XOR operation to insert stone in player's bitboard
        self.column_counts[column] += 1  # update number of stones in column

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

        if self.is_winner_cache[player] is not None:
            return self.is_winner_cache[player]

        for d in self.win_conditions:
            bb = self.bitboard[player]
            for i in range(1, self.win_length):
                bb &= self.bitboard[player] >> (i * d)
            if bb != 0:
                self.is_winner_cache[player] = True
                return True
        self.is_winner_cache[player] = False
        return False

    def is_draw(self) -> bool:
        """Is the game a draw?

        :return: True if the game is drawn, else False.
        """
        return not (
            self.get_moves()
            or self.is_winner(self.player)
            or self.is_winner(self.player ^ 1)
        )

    def is_game_over(self) -> bool:
        """Is the game over?

        :return: True if the game is over, else False.
        """
        return self.is_winner(self.player) or self.is_winner(self.player ^ 1) or not self.get_moves()

    def get_moves(self) -> List[int]:
        """Get a list of available moves.

        :return: A list of action indexes.
        """
        if self.is_winner(self.player) or self.is_winner(self.player ^ 1):
            return []  # if terminal state, return empty list

        return [
            i
            for i in range(self.board_width)
            if self.column_counts[i] < self.board_height
        ]

    def get_action_mask(self) -> List[int]:
        """Fetch a mask of valid actions

        :return: A list of ints where 1 if valid move else 0.
        """
        return [1 if self.column_counts[i] < self.board_height else 0 for i in range(self.board_width)]

    def is_valid_move(self, column: int) -> bool:
        """Check if column is full.

        :param column: The column to check
        :return: True if it is a valid move, else False.
        """
        return self.empty_indexes[column] != self.top_row[column]

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
