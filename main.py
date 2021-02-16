from env.env import DrivingEnv
import time

config_path = "env/config/scenarioConfig.yaml"
env = DrivingEnv(configpath=config_path)
env.make(scenario_name="autobahn-v0")

for i in range(50):
    env.render()
    time.sleep(0.1)
    env.step()
