import time
from stable_baselines3 import PPO
from safe_energy_nav_env import SafeEnergyNavEnv

env = SafeEnergyNavEnv(
    use_safety=True,
    use_energy=False,
    render_mode="human"
)

model = PPO.load("ppo_safe.zip")

obs, _ = env.reset()

while True:
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        env.render()
        time.sleep(0.03)

        done = terminated or truncated

    obs, _ = env.reset()
