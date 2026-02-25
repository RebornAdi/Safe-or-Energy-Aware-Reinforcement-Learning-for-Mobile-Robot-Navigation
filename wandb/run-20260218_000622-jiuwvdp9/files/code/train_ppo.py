import torch
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from safe_energy_nav_env import SafeEnergyNavEnv

# ---------------- SELECT AGENT ----------------
USE_SAFETY = 
USE_ENERGY = True
RUN_NAME = "ppo_energy"

# ---------------- INIT WANDB ----------------
run = wandb.init(
    project="Safe-Energy-RL",
    name=RUN_NAME,
    config={
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "n_steps": 4096,
        "batch_size": 256,
        "gamma": 0.99,
        "use_safety": USE_SAFETY,
        "use_energy": USE_ENERGY,
    },
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

def make_env():
    env = SafeEnergyNavEnv(use_safety=USE_SAFETY, use_energy=USE_ENERGY)
    return Monitor(env)

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    device="cuda",
    tensorboard_log=f"runs/{run.id}",
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1
)

model.learn(
    total_timesteps=200_000,
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
)

model.save(RUN_NAME)
run.finish()
