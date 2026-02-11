from stable_baselines3 import PPO
from safe_energy_nav_env import SafeEnergyNavEnv

env = SafeEnergyNavEnv(
    use_safety=True,
    use_energy=True,
    render_mode="human"
)

model = PPO.load("ppo_safe")

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()
