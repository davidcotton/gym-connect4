import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Connect4Env-v0',
    entry_point='gym-connect4.envs:Connect4Env',
)
