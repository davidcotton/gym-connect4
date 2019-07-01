# 2 player Connect4 OpenAI Gym environment
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
    import gym_connnect4

Make the Connect4 environment

    env = gym.make('gym-connect4')

Use with 2 OpenAI Gym compatible agents

    agents = [agent1(), agent2()]
    obs = env.reset()
    game_over = False
    while not game_over:
        for player, agent in enumerate(agents):
            next_obs, reward, game_over, info = env.step(action)
            obs = next_obs

## Credits
Based on [BielStela's Connect4 game](https://github.com/BielStela/connect-four), I just made it into a Gym environment.
