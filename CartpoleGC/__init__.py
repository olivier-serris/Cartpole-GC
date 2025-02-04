import gymnasium as gym
from CartpoleGC.gc_cartpole import GC_ContinuousCartpoleVector, GC_ContinuousCartpole


gym.register(
    id=f"GC-Continuous-Cartpole-v0",
    entry_point="CartpoleGC:GC_ContinuousCartpole",
    vector_entry_point="CartpoleGC:GC_ContinuousCartpoleVector",
    max_episode_steps=500,
)
