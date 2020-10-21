from gym.envs.registration import register
from .walker2d_mod import Walker2dEnv_modified

register(id='Walker2dmodified-v1', max_episode_steps=2000, entry_point='classireg.objectives.walker_env.walker2d_mod:Walker2dEnv_modified')