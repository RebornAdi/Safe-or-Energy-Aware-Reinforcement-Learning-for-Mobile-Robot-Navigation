import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from safe_energy_nav_env import SafeEnergyNavEnv

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# ---------------- SELECT AGENT ----------------
# Baseline: use_safety=False, use_energy=False
# Safe:     use_safety=True,  use_energy=False
# Energy:   use_safety=False, use_energy=True
# OURS:     use_safety=True,  use_energy=True

env = SafeEnergyNavEnv(use_safety=True, use_energy=True)

check_env(env)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    device="cuda",                 # 🔥 GPU ENABLED
    verbose=1,
    policy_kwargs=dict(net_arch=[256, 256])
)

model.learn(total_timesteps=500_000)
model.save("ppo_ours")

env.close()
