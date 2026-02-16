import numpy as np
from stable_baselines3 import PPO
from safe_energy_nav_env import SafeEnergyNavEnv

def evaluate(model_path, use_safety, use_energy, episodes=50):

    env = SafeEnergyNavEnv(use_safety=use_safety, use_energy=use_energy)
    model = PPO.load(model_path)

    success = 0
    collisions = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                if np.linalg.norm(env.goal - env.position) < 0.3:
                    success += 1
                if np.linalg.norm(env.obstacle - env.position) < env.collision_distance:
                    collisions += 1

            done = terminated or truncated

    print("Success Rate:", success / episodes)
    print("Collisions:", collisions)

if __name__ == "__main__":
    evaluate("ppo_baseline", False, False)
