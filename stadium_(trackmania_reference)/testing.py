# Third party dependencies
from repym import Repym
from stable_baselines3 import PPO

env = Repym(to_print=True)
episodes = 1
models_dir = "models"
model_path = f"{models_dir}/1734357217/70000.zip"
model = PPO.load(model_path, env=env)

for episode in range(episodes):
    done = False
    obs, info = env.reset()
    while not done:
        # raw_input = input("Waiting...\t(0 - Buy\t1 - Sell\t2 - Noop)")
        # if raw_input == "":
        #     action = 2
        # else:
        #     action = int(raw_input)

        action, _states = model.predict(obs)
        print("action", action)
        obs, reward, done, truncated, infos = env.step(action)
        print("reward", reward)
        env.render()
