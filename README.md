# 2-player Connect4 gym environment
OpenAI Gym environment for the classic board game - Connect4. 
Designed for adversarial reinforcement learning, requires two agents to play. 

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

    agents = [Agent1(), Agent2()]
    obs = env.reset()
    game_over = False
    while not game_over:
        action_dict = {}
        for agent_id, agent in enumerate(agents):
            action = env.action_space.sample()
            action_dict[agent_id] = action
        
        obs, reward, game_over, info = env.step(action_dict)
        env.render()

As Connect4 is an alternating turn game, the env is structured so both agents receive an obs at every time step, 
even on their opponents turn. However, during an opponents turn, the only legal move for an agent is a special "pass" action. 
The "pass" action is encoded to be the last action in both the action space and the action mask. 

The action space is each column, [0-6] (for default width 7 Connect4) plus an extra "pass" action. e.g.

    gym.spaces.Discrete(8)

The observation space is a dictionary containing: the action mask, the game board, your player ID, and the current player's ID, e.g.

    gym.spaces.Dict({
        'action_mask': gym.spaces.Box(low=0, high=1, shape=(self.game.board_width + 1,), dtype=np.uint8),
        'board': gym.spaces.Box(low=0, high=2, shape=(self.game.board_height, self.game.board_width), dtype=np.uint8),
        'current_player': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
        'player_id': gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
    })

These can be accessed like a normal Python dict, e.g.

    obs, reward, game_over, info = env.step(action_dict)
    action_mask = obs['action_mask']
    
    >>> numpy.ndarray([1, 1, 1, 0, 1, 1, 1, 0])


## Additional Methods
I've added some additional helper methods and properties to the environment:

### Winner
Get the game winner,

    winner = env.winner
    >>> 1

Where:
- None: game in progress
- 0: draw
- 1: player 1 is winner
- 2: player 2 is winner


## Credits
Based on [BielStela's Connect4 game](https://github.com/BielStela/connect-four), I just made it into a Gym environment.
