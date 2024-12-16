# Third party dependencies
from repym import Repym
from stable_baselines3.common.env_checker import check_env

env = Repym()

check_env(env)
