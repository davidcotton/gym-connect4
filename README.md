# 2-player Connect4 gym environment
OpenAI Gym environment for the classic board game - Connect4. 
Designed for adversarial reinforcement learning, requires two agents to play. 

## Requirements
- Python 3

## Installation
Use pip to install the git repo

    pip install -e git+ssh://git@github.com/davidcotton/connect4@master#egg=connect4

## Usage
Import Gym and the Connect4 environment

    import gym
    import gym_connect4

Build a new Connect4 environment

    env = gym.make('Connect4Env-v0')

Use with 2 OpenAI Gym compatible agents using the usual Gym methods (`reset`, `step`, `render`). 
The action space is each column, [0-6]. 

    agents = [0, 1]
    obs = env.reset()
    game_over = False
    while not game_over:
        for player, agent in enumerate(agents):
            action = env.action_space.sample()
            obs, reward, game_over, info = env.step(action)
            env.render()

As Connect4 is a 2-player alternating-turn game, after each player makes an action a new observation is available. 

## Additional Methods
I've added some additional helper methods and properties to the environment:

### Valid Moves
Get a set of valid moves available,

    env.valid_actions()
    >>> Set(0, 4, 6)

### Winner
Get the game winner,

    winner = env.winner
    >>> 1

Where:
- None: game in progress
- 0: draw
- 1: player 1 is winner
- 2: player 2 is winner

### Time
Get the number of moves made this game,
 
    time = env.time
    >>> 14

## Credits
Based on [BielStela's Connect4 game](https://github.com/BielStela/connect-four), I just made it into a Gym environment.
