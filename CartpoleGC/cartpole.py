"""
##Update code of gymnasium : 


Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing import Optional, Tuple, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space


class ContinuousCartPoleVectorEnv(VectorEnv):
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 50,
    }

    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 500,
        render_mode: Optional[str] = None,
        sutton_barto_reward: bool = False,
        gravity=9.8,
        masscart=1.0,
        masspole=0.1,
        length=0.5,
        force_mag=10.0,
        tau=0.02,
        kinematics_integrator="euler",
        x_threshold=2.4,
        x_invariance=False,
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masspole + self.masscart
        self.length = length  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag
        self.tau = tau  # seconds between state updates
        self.kinematics_integrator = kinematics_integrator
        self.x_invariance = x_invariance

        self.state = None

        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.prev_done = np.zeros(num_envs, dtype=np.bool_)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = x_threshold

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                self.theta_threshold_radians * 2,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.low = np.array([-x_threshold, -0.05, -0.05, -0.05])
        self.high = np.array([x_threshold, 0.05, 0.05, 0.05])

        self.single_action_space = spaces.Box(low=-1, high=1)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.screen_width = 600
        self.screen_height = 400
        self.clock = None
        self.screens = None
        self.surf = None

        self.steps_beyond_terminated = None

        self.goal = None

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        assert self.action_space.contains(
            actions
        ), f"{actions!r} ({type(actions)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.state
        force = (actions * self.force_mag).reshape(-1)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.stack((x, x_dot, theta, theta_dot))

        # terminal only when the pole is falling:
        terminated = (theta < -self.theta_threshold_radians) | (
            theta > self.theta_threshold_radians
        )
        # terminal when the agents gets out of x bounds:
        if not self.x_invariance:
            terminated = terminated | (x < -self.x_threshold) | (x > self.x_threshold)
        terminated = np.array(terminated)

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        if self._sutton_barto_reward:
            reward = -np.array(terminated, dtype=np.float32)
        else:
            reward = np.ones_like(terminated, dtype=np.float32)

        # Reset all environments which terminated or were truncated in the last step
        random_state = self.np_random.uniform(
            low=self.low, high=self.high, size=(self.prev_done.sum(), 4)
        )
        self.state[:, self.prev_done] = random_state.T
        self.steps[self.prev_done] = 0
        reward[self.prev_done] = 0.0
        terminated[self.prev_done] = False
        truncated[self.prev_done] = False

        self.prev_done = np.logical_or(terminated, truncated)

        if self.render_mode == "human":
            self.render()
        return (
            self.state.T.astype(np.float32),
            reward,
            terminated,
            truncated,
            {"reward_survive": np.logical_not(terminated).reshape(-1, 1)},
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # -0.05 and 0.05 is the default low and high bounds
        random_state = self.np_random.uniform(
            low=self.low, high=self.high, size=(self.num_envs, 4)
        )
        self.state = random_state.T
        if options and "reset_pos" in options:
            self.state[0, :] = np.repeat(
                options["reset_pos"].reshape(-1, 1), self.num_envs, axis=0
            ).T

        self.steps_beyond_terminated = None
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_done = np.zeros(self.num_envs, dtype=np.bool_)

        if self.render_mode == "human":
            self.render()

        return self.state.T.astype(np.float32), {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make_vec("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            )

        if self.screens is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screens = [
                    pygame.display.set_mode((self.screen_width, self.screen_height))
                    for _ in range(self.num_envs)
                ]
            else:  # mode == "rgb_array"
                pygame.init()

                self.screens = [
                    pygame.Surface((self.screen_width, self.screen_height))
                    for _ in range(self.num_envs)
                ]
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            raise ValueError(
                "Cartpole's state is None, it probably hasn't be reset yet."
            )

        for x, screen in zip(self.state.T, self.screens):
            assert isinstance(x, np.ndarray) and x.shape == (4,)

            self.surf = pygame.Surface((self.screen_width, self.screen_height))
            self.surf.fill((255, 255, 255))

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
            carty = 100  # TOP OF CART
            cart_coords = [(l, b), (l, t), (r, t), (r, b)]
            cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
            gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
            gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

            if self.goal is not None:
                width_scale = 0.05 * self.screen_width / self.x_threshold
                l, r, t, b = (
                    -width_scale / 2,
                    width_scale / 2,
                    cartheight / 2,
                    -cartheight / 2,
                )
                goal_x = self.goal * scale + self.screen_width / 2.0  # MIDDLE OF GOAL
                goal_coords = [(l, b), (l, t), (r, t), (r, b)]
                goal_coords = [(c[0] + goal_x, c[1] + carty) for c in goal_coords]
                gfxdraw.aapolygon(self.surf, goal_coords, (204, 0, 0))
                gfxdraw.filled_polygon(self.surf, goal_coords, (204, 0, 0))

            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )

            pole_coords = []
            for coord in [(l, b), (l, t), (r, t), (r, b)]:
                coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
                coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
                pole_coords.append(coord)
            gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
            gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

            gfxdraw.aacircle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )
            gfxdraw.filled_circle(
                self.surf,
                int(cartx),
                int(carty + axleoffset),
                int(polewidth / 2),
                (129, 132, 203),
            )

            gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

            self.surf = pygame.transform.flip(self.surf, False, True)
            screen.blit(self.surf, (0, 0))
            if self.render_mode == "human":
                pygame.event.pump()
                self.clock.tick(self.metadata["render_fps"])
                pygame.display.flip()

        return [
            np.transpose(np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2))
            for screen in self.screens
        ]

    def close(self):
        if self.screens is not None:
            import pygame

            pygame.quit()

    def is_terminated(self, observation):
        x = observation[0]
        theta = observation[2]

        x_term = np.logical_or(x < -self.x_threshold, x > self.x_threshold)
        thetha_term = np.logical_or(
            theta < -self.theta_threshold_radians, theta > self.theta_threshold_radians
        )
        return np.logical_or(x_term, thetha_term)
