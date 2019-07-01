from typing import Set

import gym
from gym import spaces
import numpy as np


BOARD_WIDTH = 7
BOARD_HEIGHT = 6


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

    def step(self, action):
        if not self.game_over:
            i = self._drop(self.player, action)
            reward = -1.0 if i < 0 else 0.0  # negative reward if invalid action (column full)
            self._winning_check(i, action)

        if self.game_over:
            reward = 1.0 if self.player == self.winner else -1.0

        if np.all(self.board):  # check if board full
            self.game_over = True
            self.winner = 0  # there was no winner

        info = {}

        self.player = 1 if self.player == 2 else 2
        self.time += 1

        return self._get_state(), reward, self.game_over, info

    def _drop(self, player, column):
        """
        Drops a number (same as player) in the column specified
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

    def valid_moves(self) -> Set[int]:
        valid_moves = set()
        for column in range(self.board.shape[1]):
            column_vec = self.board[:, column]
            nonzero = np.count_nonzero(column_vec)
            if nonzero < self.board.shape[0]:
                valid_moves.add(column)
        return valid_moves

    def _winning_check(self, i, j) -> None:
        """
        Checks if there is four equal numbers in every
        row, column and diagonal of the matrix
        """
        all_arr = []
        all_arr.extend(self._get_axes(self.board, i, j))
        all_arr.extend(self._get_diagonals(self.board, i, j))

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
        print(self._get_state(), '\n')
        if self.winner:
            print('winner:', self.winner)


