# we should be able to use a continuous space - we program the actions to work in ranges but the range is dictated by the algo
# we can also start with a discrete one and see if there's anything promising there
# from gym MultiDiscrete: A list of possible actions, where each timestep only one action of each discrete set can be used.
# discrete sell actions: sell 25%, sell 50%, sell 100%
# discrete buy actions: buy with 5% of balance, buy with 10%, double-down (maybe)
# we can use A2C
# we can use PPO
# we can use RecurrentPPO
# we can use TRPO
# we can use Maskable PPO

# we will use A2C from stablebaselines
# we will have a vectorized environment so we can have an environment that shows the increments file and another where we show the current positions.


# https://github.com/PaulSwenson2/ReinforcementLearningProjects/blob/main/BasicEnvironment/BasicEnvironment.py
# https://huggingface.co/learn/deep-rl-course/unitbonus3/model-based
# https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/trpo/trpo.py
# https://stable-baselines3.readthedocs.io/en/master/guide/sb3_contrib.html#sb3-contrib
# https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html
# https://medium.com/@paulswenson2/an-introduction-to-building-custom-reinforcement-learning-environment-using-openai-gym-d8a5e7cf07ea
# https://medium.com/analytics-vidhya/reinforcement-learning-in-continuous-action-spaces-ddpg-bbd64aa5434
# https://www.backtrader.com/docu/order-creation-execution/bracket/bracket/#sample-code
# https://dl.acm.org/doi/fullHtml/10.1145/3508546.3508598
# https://openscholarship.wustl.edu/cgi/viewcontent.cgi?article=1017&context=eseundergraduate_research
# https://www.phind.com/search?cache=oq60dpo12z2a62oy5xc7jwyi
# https://www.tensorflow.org/guide/keras/working_with_rnns
# https://keras.io/examples/rl/actor_critic_cartpole/
# https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
# https://github.com/araffin/rl-tutorial-jnrr19/blob/sb3/1_getting_started.ipynb
# https://optuna.org/
# https://github.com/DLR-RM/rl-baselines3-zoo?tab=readme-ov-file
# https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vec-env


# Start
# https://pythonprogramming.net/introduction-reinforcement-learning-stable-baselines-3-tutorial/
# import gym
# from stable_baselines3 import A2C

# env = gym.make("LunarLander-v2")  # continuous: LunarLanderContinuous-v2
# env.reset()

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1000000)

# episodes = 10000000000

# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         ret = env.render(ansi)
#         print(rewards)

# # Save a checkpoint every 1000 steps
# checkpoint_callback = CheckpointCallback(
#   save_freq=1000,
#   save_path="./logs/",
#   name_prefix="rl_model",
#   save_replay_buffer=True,
#   save_vecnormalize=True,
# )
# Display progress bar using the progress bar callback
# # this is equivalent to model.learn(100_000, callback=ProgressBarCallback())
# model.learn(100_000, progress_bar=True)


# model = A2C("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
# model.learn(total_timesteps=10_000)


# class FigureRecorderCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super().__init__(verbose)

# Third party dependencies
# from repym import Repym

# #     def _on_step(self):
# #         # Plot values (here a random variable)
# #         figure = plt.figure()
# #         figure.add_subplot().plot(np.random.random(3))
# #         # Close the figure after logging it
# #         self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
# #         plt.close()
# #         return True
# from stable_baselines3.common.env_checker import check_env

# env = Repym()
# # It will check your custom environment and output additional warnings if needed
# check_env(env)

# Standard Library
import os
import time

# Third party dependencies
import gymnasium as gym
import torch as th
from repym import Repym
from rllte.xplore.reward import RND
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


class RLeXploreWithOnPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3.
    """

    def __init__(self, irs, verbose=0):
        super(RLeXploreWithOnPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        observations = self.locals["obs_tensor"]
        device = observations.device
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
        self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:
        # ===================== compute the intrinsic rewards ===================== #
        # prepare the data samples
        obs = th.as_tensor(self.buffer.observations)
        # get the new observations
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)
        # print(obs.shape, actions.shape, rewards.shape, dones.shape, obs.shape)
        # compute the intrinsic rewards
        intrinsic_rewards = irs.compute(
            samples=dict(
                observations=obs,
                actions=actions,
                rewards=rewards,
                terminateds=dones,
                truncateds=dones,
                next_observations=new_obs,
            ),
            sync=True,
        )
        # add the intrinsic rewards to the buffer
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #


class RLeXploreWithOffPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and off-policy algorithms from SB3.
    """

    def __init__(self, irs, verbose=0):
        super(RLeXploreWithOffPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.replay_buffer

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        device = self.irs.device
        obs = th.as_tensor(self.locals["self"]._last_obs, device=device)
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_obs = th.as_tensor(self.locals["new_obs"], device=device)

        # ===================== watch the interaction ===================== #
        self.irs.watch(obs, actions, rewards, dones, dones, next_obs)
        # ===================== watch the interaction ===================== #

        # ===================== compute the intrinsic rewards ===================== #
        intrinsic_rewards = irs.compute(
            samples={
                "observations": obs.unsqueeze(0),
                "actions": actions.unsqueeze(0),
                "rewards": rewards.unsqueeze(0),
                "terminateds": dones.unsqueeze(0),
                "truncateds": dones.unsqueeze(0),
                "next_observations": next_obs.unsqueeze(0),
            },
            sync=False,
        )
        # ===================== compute the intrinsic rewards ===================== #

        try:
            # add the intrinsic rewards to the original rewards
            self.locals["rewards"] += intrinsic_rewards.cpu().numpy().squeeze()
            # update the intrinsic reward module
            replay_data = self.buffer.sample(batch_size=self.irs.batch_size)
            self.irs.update(
                samples={
                    "observations": th.as_tensor(replay_data.observations)
                    .unsqueeze(1)
                    .to(device),  # (n_steps, n_envs, *obs_shape)
                    "actions": th.as_tensor(replay_data.actions).unsqueeze(1).to(device),
                    "rewards": th.as_tensor(replay_data.rewards).to(device),
                    "terminateds": th.as_tensor(replay_data.dones).to(device),
                    "truncateds": th.as_tensor(replay_data.dones).to(device),
                    "next_observations": th.as_tensor(replay_data.next_observations).unsqueeze(1).to(device),
                }
            )
        except Exception as e:
            print(e)
            pass

        return True

    def _on_rollout_end(self) -> None:
        pass


# # Parallel environments
device = "cpu"
n_envs = 8
env = Repym
envs = make_vec_env(env, n_envs=n_envs)

# ===================== build the reward ===================== #
irs = RND(envs, device=device)
# ===================== build the reward ===================== #

model = PPO("MlpPolicy", envs, verbose=1, device=device, tensorboard_log=logdir)
TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    # model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=RLeXploreWithOnPolicyRL(irs))

    model.save(f"{models_dir}/{TIMESTEPS*iters}")
