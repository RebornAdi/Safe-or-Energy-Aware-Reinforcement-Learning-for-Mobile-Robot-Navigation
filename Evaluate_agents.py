import numpy as np
from stable_baselines3 import PPO
from safe_energy_nav_env import SafeEnergyNavEnv

def evaluate_model(model_path, use_safety, use_energy, episodes=50):
    env = SafeEnergyNavEnv(
        use_safety=use_safety,
        use_energy=use_energy
    )

    model = PPO.load(model_path)

    success_count = 0
    collision_count = 0
    total_energy = 0
    total_path = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        prev_position = env.position.copy()

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            # Energy
            total_energy += (env.v**2 + env.omega**2)

            # Path length
            step_dist = np.linalg.norm(env.position - prev_position)
            total_path += step_dist
            prev_position = env.position.copy()

            if terminated and np.linalg.norm(env.goal - env.position) < 0.3:
                success_count += 1

            if terminated and np.linalg.norm(env.obstacle - env.position) < env.collision_distance:
                collision_count += 1

            done = terminated or truncated

    env.close()

    print(f"\nResults for {model_path}")
    print(f"Success Rate: {success_count/episodes:.2f}")
    print(f"Collisions: {collision_count}")
    print(f"Average Energy: {total_energy/episodes:.2f}")
    print(f"Average Path Length: {total_path/episodes:.2f}")
    print("-"*40)


if __name__ == "__main__":

    evaluate_model("ppo_baseline", False, False)
    evaluate_model("ppo_safe", True, False)
    evaluate_model("ppo_energy", False, True)
    evaluate_model("ppo_ours", True, True)
# ---------------- SELECT AGENT ----------------
# Baseline: use_safety=False, use_energy=False
# Safe:     use_safety=True,  use_energy=False
# Energy:   use_safety=False, use_energy=True
# OURS:     use_safety=True,  use_energy=True