from .connect4_env import Connect4


class TestConnect4:
    def test_move_clears_is_winner_cache(self):
        game = Connect4()
        game.move(3)
        assert game.is_winner_cache == [None, None]

    def test_move_increments_empty_indexes(self):
        game = Connect4()
        assert game.empty_indexes == [0, 7, 14, 21, 28, 35, 42]

        game.move(0)
        assert game.empty_indexes == [1, 7, 14, 21, 28, 35, 42]

        game.move(3)
        assert game.empty_indexes == [1, 7, 14, 22, 28, 35, 42]

        game.move(3)
        assert game.empty_indexes == [1, 7, 14, 23, 28, 35, 42]

    def test_move_increments_player(self):
        game = Connect4()
        assert game.player == 1  # game starts with current player = 1

        game.move(0)
        assert game.player == 0

        game.move(0)
        assert game.player == 1

    def test_move_increments_bitboard(self):
        game = Connect4()
        game.move(0)
        assert game.bitboard == [1, 0]

        game.move(0)
        assert game.bitboard == [1, 2]

    def test_move_increments_column_counts(self):
        game = Connect4()

        game.move(0)
        assert game.column_counts == [1, 0, 0, 0, 0, 0, 0]

        game.move(1)
        assert game.column_counts == [1, 1, 0, 0, 0, 0, 0]

        game.move(1)
        assert game.column_counts == [1, 2, 0, 0, 0, 0, 0]

    def test_get_reward_on_winner(self):
        game = Connect4()
        game.move(0)
        game.move(1)
        game.move(0)
        game.move(1)
        game.move(0)
        game.move(1)
        assert game.get_reward() == 0.0
        assert game.get_reward(0) == 0.0
        assert game.get_reward(1) == 0.0
        game.move(0)
        assert game.get_reward() == 1.0
        assert game.get_reward(0) == 1.0
        assert game.get_reward(1) == -1.0

    def test_is_game_over_on_init(self):
        game = Connect4()
        assert not game.is_game_over()

    def test_is_game_over_on_winner(self):
        game = Connect4()
        game.move(0)
        game.move(1)
        game.move(0)
        game.move(1)
        game.move(0)
        game.move(1)
        assert not game.is_game_over()
        game.move(0)
        assert game.is_game_over()

    def test_get_moves_on_init(self):
        game = Connect4()
        assert game.get_moves() == [0, 1, 2, 3, 4, 5, 6]

    def test_get_moves_on_single_move(self):
        game = Connect4()
        game.move(1)
        assert game.get_moves() == [0, 1, 2, 3, 4, 5, 6]
        game.move(2)
        assert game.get_moves() == [0, 1, 2, 3, 4, 5, 6]
        game.move(3)
        assert game.get_moves() == [0, 1, 2, 3, 4, 5, 6]

    def test_get_moves_on_full_column(self):
        game = Connect4()
        assert game.get_moves() == [0, 1, 2, 3, 4, 5, 6]
        game.move(2)
        game.move(2)
        game.move(2)
        game.move(2)
        game.move(2)
        game.move(2)
        assert game.get_moves() == [0, 1, 3, 4, 5, 6]
        game.move(3)
        assert game.get_moves() == [0, 1, 3, 4, 5, 6]
        game.move(3)
        game.move(3)
        game.move(3)
        game.move(3)
        game.move(3)
        assert game.get_moves() == [0, 1, 4, 5, 6]

    def test_get_action_mask_creates_mask_on_init(self):
        game = Connect4()
        mask = game.get_action_mask()
        assert mask == [1, 1, 1, 1, 1, 1, 1]

    def test_get_action_mask_with_single_move(self):
        game = Connect4()
        game.move(3)
        mask = game.get_action_mask()
        assert mask == [1, 1, 1, 1, 1, 1, 1]

    def test_get_action_mask_with_full_column(self):
        game = Connect4()
        game.move(3)
        game.move(3)
        game.move(3)
        game.move(3)
        game.move(3)
        game.move(3)
        mask = game.get_action_mask()
        assert mask == [1, 1, 1, 0, 1, 1, 1]

    def test_is_valid_move_on_init(self):
        game = Connect4()
        assert game.is_valid_move(4)

    def test_is_valid_move_on_single_move(self):
        game = Connect4()
        game.move(4)
        assert game.is_valid_move(4)

    def test_is_valid_move_on_full_column(self):
        game = Connect4()
        game.move(4)
        game.move(4)
        game.move(4)
        game.move(4)
        game.move(4)
        game.move(4)
        assert not game.is_valid_move(4)
