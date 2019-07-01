from gym.envs.registration import register

register(
    id='Connect4Env-v0',
    entry_point='gym_connect4.envs:Connect4Env',
)
