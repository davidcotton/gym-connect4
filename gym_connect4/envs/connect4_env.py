from typing import Set, Tuple

import gym
from gym import spaces
import numpy as np


BOARD_WIDTH = 7
BOARD_HEIGHT = 6
REWARD_WIN = 1.0
REWARD_LOSS = -1.0


class Connect4Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(BOARD_WIDTH)
        self.observation_space = spaces.Box(low=0, high=2, shape=(BOARD_HEIGHT, BOARD_WIDTH), dtype=np.uint8)
        self.obs_type = 'image'
        self.board = np.zeros([BOARD_HEIGHT, BOARD_WIDTH], dtype=np.uint8)
        self.time = 0
        self.game_over = False
        self.player = 1
        self.winner = None

    def reset(self) -> np.ndarray:
        self.board = np.zeros([BOARD_HEIGHT, BOARD_WIDTH], dtype=np.uint8)
        self.time = 0
        self.game_over = False
        self.player = 1
        self.winner = None
        return self._get_state()

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        """Make a game action.

        Throws a ValueError if trying to drop into a full column.

        :param action: The column index to drop in.
        :return: A tuple containing the next state, reward, if the game ended and an empty info dict.
        """
        if not self.game_over:
            remaining_rows = self._drop(self.player, action)
            if remaining_rows < 0:
                raise ValueError('Invalid action, column full.')
            self._winning_check(remaining_rows, action)

        reward = 0.0
        if self.game_over:
            reward = REWARD_WIN if self.player == self.winner else REWARD_LOSS

        if np.all(self.board):  # check if board full
            self.game_over = True
            self.winner = 0  # there was no winner

        info = {}

        self.player = 1 if self.player == 2 else 2
        self.time += 1

        return self._get_state(), reward, self.game_over, info

    def _drop(self, player, column) -> int:
        """Drop a coin into the specified column.

        :param player: The player making the action.
        :param column: The column the coin is being placed.
        :return: The number of remaining rows above the placed coin.
        """
        column_vec = self.board[:, column]
        non_zero = np.where(column_vec != 0)[0]

        if non_zero.size == 0:
            # sets the stone to the last element
            i = self.board.shape[0] - 1
            self.board[i, column] = player
        else:
            # sets the stone on the last 0
            i = non_zero[0] - 1
            if i >= 0:
                self.board[i, column] = player
        return i

    def valid_actions(self) -> Set[int]:
        """Fetch a set of valid moves available.

        :return: A set of valid moves.
        """
        valid_moves = set()
        for column in range(self.board.shape[1]):
            column_vec = self.board[:, column]
            nonzero = np.count_nonzero(column_vec)
            if nonzero < self.board.shape[0]:
                valid_moves.add(column)
        return valid_moves

    def _winning_check(self, i, action) -> None:
        """Checks if there is four equal numbers in every row, column and diagonal of the matrix.

        :param i:
        :param action: The column the player dropped into.
        """
        all_arr = []
        all_arr.extend(self._get_axes(self.board, i, action))
        all_arr.extend(self._get_diagonals(self.board, i, action))

        for arr in all_arr:
            winner = self._winning_rule(arr)
            if winner:
                self.game_over = True
                self.winner = self.player

    def _winning_rule(self, arr) -> bool:
        win1rule = np.array([1, 1, 1, 1])
        win2rule = np.array([2, 2, 2, 2])
        # subarrays of len = 4
        sub_arrays = [arr[i:i + 4] for i in range(len(arr) - 3)]

        player1wins = any([np.array_equal(win1rule, sub) for sub in sub_arrays])
        player2wins = any([np.array_equal(win2rule, sub) for sub in sub_arrays])

        if player1wins or player2wins:
            return True
        else:
            return False

    def _get_diagonals(self, board, i, j) -> list:
        diags = []
        diags.append(np.diagonal(board, offset=(j - i)))
        diags.append(np.diagonal(np.rot90(board), offset=-board.shape[1] + (j + i) + 1))
        return diags

    def _get_axes(self, board, i, j) -> list:
        return [board[i, :], board[:, j]]

    def _get_state(self) -> np.ndarray:
        return self.board.copy()

    def render(self, mode='human') -> None:
        print('  1 2 3 4 5 6 7')
        print(' ---------------')
        print(self._get_state())
        print(' ---------------')
        print('  1 2 3 4 5 6 7')
        print()
        if self.winner is not None:
            print('##########')
            print('Winner:', self.winner)
            print('##########')
