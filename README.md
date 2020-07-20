# 2-player Connect4 gym environment
OpenAI Gym environment for the classic board game - Connect4. 
Designed for competitive reinforcement learning, requires two agents to play. 
Designed for speed. Pure python, but uses bitboards (bitwise operations) to try and get you the maximum simulation speed. 

## Requirements
- Python 3

## Installation
Use pip to install the git repo

    pip install -e git+ssh://git@github.com/davidcotton/gym-connect4@master#egg=gym-connect4

## Usage
Import Gym and the Connect4 environment

    import gym
    import gym_connect4

Build a new (default) Connect4 environment via the usual Gym factory method

    env = gym.make('Connect4Env-v0')

Then use similar to usual Gym workflow run the env, except that both players receive obs and generate an action each turn.

    agents = ['Agent1()', 'Agent2()']
    obses = env.reset()  # dict: {0: obs_player_1, 1: obs_player_2}
    game_over = False
    while not game_over:
        action_dict = {}
        for agent_id, agent in enumerate(agents):
            action = env.action_space.sample()
            action_dict[agent_id] = action
        
        obses, rewards, game_over, info = env.step(action_dict)
        env.render()

As Connect4 is an alternating turn game, the env is structured so both agents receive an obs at every time step, 
even on their opponents turn. However, during an opponents turn, the only legal move for an agent is a special "pass" action. 
The "pass" action is encoded to be the last action in both the action space and the action mask. 

The action space is each column, [0-6] (for default width 7 Connect4) plus an extra "pass" action. e.g.

    gym.spaces.Discrete(8)

The observation space is a dictionary containing: the action mask and the game board, e.g.

    gym.spaces.Dict({
        'action_mask': gym.spaces.Box(low=0, high=1, shape=(self.game.board_width + 1,), dtype=np.uint8),
        'board': gym.spaces.Box(low=0, high=2, shape=(self.game.board_height, self.game.board_width), dtype=np.uint8),
    })

These can be accessed like a normal Python dict, e.g.

    obs, reward, game_over, info = env.step(action_dict)
    action_mask = obs['action_mask']
    
    >>> numpy.ndarray([1, 1, 1, 0, 1, 1, 1, 0])


### Configuration
#### Custom Game Initialisation
Optionally, you can change the default environment configuration 

    env_config = {
        'board_height': 4,
        'board_width': 5,
    }
    env = gym.make('Connect4Env-v0', env_config=env_config)

The available parameters are:

| Param | Description | Default |
|-------|-------------|---------|
| board_height | The number of rows on the connect4 board | 6 |
| board_width | The number of columns on the connect4 board |  7 |
| win_length | The number of consecutive discs need to win | 4 |
| reward_win | The utility of winning | 1.0 |
| reward_lose | The utility of losing | -1.0 |
| reward_draw | The utility of a draw | 0.0 |
| reward_step | The utility of each turn | 0.0 |


#### Custom Game States
If you are using a search-based algorithm such as MCTS or Minimax, you might want the ability to initialise the a game in a custom state, e.g. part way through a game.
Rather than building the full Gym environment, you can just use the inner game class with a custom config and/or game state:

    from gym_connect4.envs.connect4_env import Connect4
    
    board = np.array([0, 0, 0, ...])
    game = Connect4(game_state={'board': board, 'player': 1})
    game.move(3)


## Credits
Based on [BielStela's Connect4 game](https://github.com/BielStela/connect-four), I just made it into a Gym environment.
