import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class SafeEnergyNavEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # =====================================================
    # INIT
    # =====================================================
    def __init__(self, use_safety=True, use_energy=True, render_mode=None):
        super().__init__()

        self.use_safety = use_safety
        self.use_energy = use_energy
        self.render_mode = render_mode

        self.world_size = 10.0
        self.dt = 0.1
        self.max_steps = 300

        self.max_v = 1.0
        self.max_omega = 1.0

        self.safe_distance = 0.8
        self.collision_distance = 0.3

        # ---------------- ACTION SPACE ----------------
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # ---------------- OBSERVATION SPACE ----------------
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([15.0, 1.0, 1.0, 15.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.fig = None
        self.ax = None

    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.goal = self._random_pos()
        self.obstacle = self._random_pos()

        while np.linalg.norm(self.goal - self.obstacle) < 1.5:
            self.obstacle = self._random_pos()

        self.position = self._random_pos()
        while (
            np.linalg.norm(self.position - self.obstacle) < 1.0 or
            np.linalg.norm(self.position - self.goal) < 1.0
        ):
            self.position = self._random_pos()

        self.heading = self.np_random.uniform(-np.pi, np.pi)
        self.v = 0.0
        self.omega = 0.0
        self.steps = 0

        return self._get_obs(), {}

    def _random_pos(self):
        return self.np_random.uniform(
            1.0, self.world_size - 1.0, size=(2,)
        ).astype(np.float32)

    # =====================================================
    # STEP
    # =====================================================
    def step(self, action):

        self.steps += 1

        # --- safety clip (numerical robustness) ---
        action = np.clip(action, -1.0, 1.0)

        # map actions
        self.v = ((action[0] + 1.0) / 2.0) * self.max_v
        self.omega = action[1] * self.max_omega

        prev_position = self.position.copy()

        # robot motion
        self.heading += self.omega * self.dt
        self.heading = (self.heading + np.pi) % (2*np.pi) - np.pi

        dx = self.v * np.cos(self.heading) * self.dt
        dy = self.v * np.sin(self.heading) * self.dt

        self.position += np.array([dx, dy])
        self.position = np.clip(self.position, 0, self.world_size)

        # distances
        dist_goal = np.linalg.norm(self.goal - self.position)
        prev_dist_goal = np.linalg.norm(self.goal - prev_position)
        dist_obs = np.linalg.norm(self.obstacle - self.position)

        reward = 0.0

        # ---------------- PROGRESS ----------------
        reward += 5.0 * (prev_dist_goal - dist_goal)

        # time penalty
        reward -= 0.01

        # ---------------- SAFETY ----------------
        if self.use_safety:
            if dist_obs < self.collision_distance:
                reward -= 100.0
            elif dist_obs < self.safe_distance:
                penalty_factor = (
                    (self.safe_distance - dist_obs)
                    / (self.safe_distance - self.collision_distance)
                )
                reward -= 10.0 * penalty_factor

        # ---------------- ENERGY ----------------
        if self.use_energy:
            reward -= 0.02 * (self.v**2 + 0.3 * self.omega**2)

        terminated = False
        truncated = False

        # goal reached
        if dist_goal < 0.3:
            reward += 200.0
            terminated = True

        # collision
        if dist_obs < self.collision_distance:
            terminated = True

        # timeout
        if self.steps >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, {}

    # =====================================================
    # OBSERVATION
    # =====================================================
    def _get_obs(self):

        vec_goal = self.goal - self.position
        heading_goal = np.arctan2(vec_goal[1], vec_goal[0])

        heading_error = heading_goal - self.heading
        heading_error = (heading_error + np.pi) % (2*np.pi) - np.pi

        return np.array([
            np.linalg.norm(vec_goal),
            np.cos(heading_error),
            np.sin(heading_error),
            np.linalg.norm(self.obstacle - self.position),
            self.v,
            self.omega
        ], dtype=np.float32)

    # =====================================================
    # RENDER
    # =====================================================
    def render(self):

        if self.render_mode != "human":
            return

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()
        self.ax.set_xlim(0, self.world_size)
        self.ax.set_ylim(0, self.world_size)

        # robot
        self.ax.plot(self.position[0], self.position[1], "bo")

        hx = self.position[0] + 0.5*np.cos(self.heading)
        hy = self.position[1] + 0.5*np.sin(self.heading)
        self.ax.plot([self.position[0], hx],
                     [self.position[1], hy], "b-")

        # goal
        self.ax.plot(self.goal[0], self.goal[1], "g*", markersize=15)

        # obstacle
        obs = plt.Circle(self.obstacle, self.collision_distance, color="r")
        self.ax.add_patch(obs)

        safe = plt.Circle(
            self.obstacle,
            self.safe_distance,
            linestyle="--",
            fill=False,
            color="r"
        )
        self.ax.add_patch(safe)

        self.ax.set_title(f"Steps: {self.steps}")
        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)