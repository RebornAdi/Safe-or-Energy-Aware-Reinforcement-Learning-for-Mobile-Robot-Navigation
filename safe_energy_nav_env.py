import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class SafeEnergyNavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, use_safety=True, use_energy=True, render_mode=None):
        super().__init__()

        self.use_safety = use_safety
        self.use_energy = use_energy
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1.0,high=1.0,shape=(2,),dtype=np.float32)


        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.pi, 0.0, 0.0, -1.0]),
            high=np.array([15.0, np.pi, 15.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.world_size = 10.0
        self.dt = 0.1
        self.max_steps = 300

        self.safe_distance = 0.6
        self.collision_distance = 0.25

        self.fig = None
        self.ax = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.position = np.array([1.0, 1.0], dtype=np.float32)
        self.heading = 0.0
        self.v = 0.0
        self.omega = 0.0

        self.goal = np.array([8.5, 8.5], dtype=np.float32)
        self.obstacle = np.array([4.5, 4.5], dtype=np.float32)

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        self.v = float(np.clip(action[0], 0.0, 1.0))
        self.omega = float(np.clip(action[1], -1.0, 1.0))

        self.heading += self.omega * self.dt
        dx = self.v * np.cos(self.heading) * self.dt
        dy = self.v * np.sin(self.heading) * self.dt
        self.position += np.array([dx, dy])

        dist_goal = np.linalg.norm(self.goal - self.position)
        dist_obs = np.linalg.norm(self.obstacle - self.position)

        reward = -dist_goal

        if self.use_safety:
            if dist_obs < self.collision_distance:
                reward -= 50.0
            elif dist_obs < self.safe_distance:
                reward -= 5.0 * (self.safe_distance - dist_obs)

        if self.use_energy:
            reward -= 0.5 * (self.v ** 2 + self.omega ** 2)

        terminated = False
        truncated = False

        if dist_goal < 0.3:
            reward += 100.0
            terminated = True

        if dist_obs < self.collision_distance:
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), float(reward), terminated, truncated, {}

    def _get_obs(self):
        vec_goal = self.goal - self.position
        heading_goal = np.arctan2(vec_goal[1], vec_goal[0])
        heading_error = heading_goal - self.heading

        return np.array([
            np.linalg.norm(vec_goal),
            heading_error,
            np.linalg.norm(self.obstacle - self.position),
            self.v,
            self.omega
        ], dtype=np.float32)

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()
        self.ax.set_xlim(0, self.world_size)
        self.ax.set_ylim(0, self.world_size)

        self.ax.plot(self.position[0], self.position[1], "bo", label="Robot")

        hx = self.position[0] + 0.5 * np.cos(self.heading)
        hy = self.position[1] + 0.5 * np.sin(self.heading)
        self.ax.plot([self.position[0], hx], [self.position[1], hy], "b-")

        self.ax.plot(self.goal[0], self.goal[1], "g*", markersize=15, label="Goal")

        obs = plt.Circle(self.obstacle, self.collision_distance, color="r")
        self.ax.add_patch(obs)

        safe = plt.Circle(
            self.obstacle, self.safe_distance,
            color="r", linestyle="--", fill=False
        )
        self.ax.add_patch(safe)

        self.ax.set_title("Safe & Energy-Aware RL Navigation")
        self.ax.legend()
        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)
