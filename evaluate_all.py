import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from safe_energy_nav_env import SafeEnergyNavEnv


MODELS = [
    ("Baseline", "ppo_baseline.zip", False, False),
    ("Safe", "ppo_safe.zip", True, False),
    ("Energy", "ppo_energy.zip", False, True),
    ("Ours", "ppo_ours.zip", True, True),
]

EPISODES = 100


def evaluate(name, path, use_safety, use_energy):

    env = SafeEnergyNavEnv(
        use_safety=use_safety,
        use_energy=use_energy
    )

    model = PPO.load(path)

    success = 0
    collisions = 0
    total_energy = 0
    total_path = 0
    total_steps = 0

    for _ in range(EPISODES):

        obs, _ = env.reset()
        done = False
        prev_pos = env.position.copy()

        while not done:

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            # energy usage
            total_energy += env.v**2 + env.omega**2

            # path length
            total_path += np.linalg.norm(env.position - prev_pos)
            prev_pos = env.position.copy()

            total_steps += 1

            if terminated:
                if np.linalg.norm(env.goal - env.position) < 0.3:
                    success += 1

                if np.linalg.norm(env.obstacle - env.position) < env.collision_distance:
                    collisions += 1

            done = terminated or truncated

    env.close()

    return {
        "Model": name,
        "SuccessRate": success / EPISODES,
        "Collisions": collisions,
        "AvgEnergy": total_energy / EPISODES,
        "AvgPathLength": total_path / EPISODES,
        "AvgSteps": total_steps / EPISODES,
    }


results = []

for m in MODELS:
    print("Evaluating:", m[0])
    results.append(evaluate(*m))

df = pd.DataFrame(results)

print("\n===== RESULTS =====")
print(df)

df.to_csv("results.csv", index=False)
