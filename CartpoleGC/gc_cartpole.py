from typing import Any
from CartpoleGC.cartpole import ContinuousCartPoleVectorEnv
from gymnasium.vector import VectorEnv
import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space


class GC_ContinuousCartpoleVector(VectorEnv):
    def __init__(
        self,
        num_envs=1,
        max_episode_steps=500,
        render_mode=None,
        x_threshold=2.4,
        goal_reached_threshold=0.05,
        x_invariance=False,
        **kwargs
    ) -> None:
        self.num_envs = num_envs
        self.cartpole = ContinuousCartPoleVectorEnv(
            num_envs=num_envs,
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
            x_threshold=x_threshold,
            x_invariance=x_invariance,
            **kwargs
        )
        self.action_space = self.cartpole.action_space
        self.single_action_space = self.cartpole.single_action_space
        self.goalspace = gym.spaces.Box(low=-x_threshold, high=x_threshold)
        self.single_observation_space = gym.spaces.Dict(
            {
                "observation": self.cartpole.observation_space,
                "achieved_goal": self.goalspace,
                "desired_goal": self.goalspace,
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                "observation": self.cartpole.observation_space,
                "achieved_goal": batch_space(self.goalspace, self.num_envs),
                "desired_goal": batch_space(self.goalspace, self.num_envs),
            }
        )
        n_eval = 6
        self.eval_options = {
            "reset_pos": np.array([0]).repeat(n_eval),
            "goal_pos": np.linspace(-x_threshold + 0.20, x_threshold - 0.20, n_eval),
        }
        self.goal_reached_threshold = goal_reached_threshold
        self.x_invariance = x_invariance

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        prev_done = np.copy(self.cartpole.prev_done)
        obs, reward, terminated, truncated, info = self.cartpole.step(actions)
        if prev_done.any():
            self.desired_goal[prev_done] = self.select_goal(obs)[prev_done]
        info["cartpole_rew"] = reward
        gc_obs = self.to_gc_obs(observation=obs, desired_goal=self.desired_goal)
        success = self.goal_reached(gc_obs, self.goal_reached_threshold)
        info["success"] = success
        reward = success

        return (gc_obs, reward, terminated, truncated, info)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:

        obs, info = self.cartpole.reset(seed=seed, options=options)
        if options and "goal_pos" in options:
            self.desired_goal = np.repeat(
                options["goal_pos"].reshape(-1, 1), repeats=self.num_envs, axis=0
            )
        else:
            self.desired_goal = self.select_goal(obs)
        self.cartpole.goal = self.desired_goal
        return self.to_gc_obs(observation=obs, desired_goal=self.desired_goal), info

    def goal_reached(self, observation, goal_reached_threshold):

        # if self.x_invariance:
        #     return np.abs(observation["desired_goal"]) < goal_reached_threshold
        # else:
        #     return (
        #         np.abs(observation["achieved_goal"] - observation["desired_goal"])
        #         < goal_reached_threshold
        #     )
        return (
            np.abs(observation["achieved_goal"] - observation["desired_goal"])
            < goal_reached_threshold
        )

    def select_goal(self, obs):

        position = self.np_random.uniform(
            self.goalspace.low, self.goalspace.high, self.num_envs
        )
        if self.x_invariance:

            position = (
                self.cartpole.state.T[:, 0]
                + self.np_random.uniform(
                    -self.x_threshold, self.x_threshold, self.num_envs
                )
                * 1.5
            )
        return position.reshape(-1, 1)

    def to_gc_obs(self, observation, desired_goal):
        return {
            "observation": observation[:, 1:] if self.x_invariance else observation,
            "achieved_goal": observation[:, 0].reshape(self.num_envs, 1),
            "desired_goal": desired_goal.reshape(self.num_envs, 1),
        }

    @property
    def x_threshold(self):
        return self.cartpole.x_threshold

    @property
    def goal_transfo(self):
        if self.x_invariance:
            return lambda ag, dg: dg - ag
        else:
            return None


class GC_ContinuousCartpole(gym.Env):
    def __init__(self, **kwargs) -> None:
        self.vector_cartpole = GC_ContinuousCartpoleVector(num_envs=1, **kwargs)
        self.observation_space = self.vector_cartpole.observation_space
        self.action_space = self.vector_cartpole.action_space

    def step(self, action: Any):
        return self.vector_cartpole.step(action)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        return self.vector_cartpole.reset(seed=seed, options=options)

    def get_wrapper_attr(self, name: str) -> Any:
        return self.vector_cartpole.__getattribute__(name)
