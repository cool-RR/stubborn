# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

from typing import Optional, Mapping

import ray.rllib.env.multi_agent_env
import ray.rllib.utils.spaces
import gym
import gym.spaces
import numpy as np

from stubborn import county
from .stubborn_state import StubbornState
from .stubborn_config import StubbornConfig



class StubbornEnv(county.BaseEnv):

    def __init__(self, config: Optional[Mapping] = None,) -> None:
        ray.rllib.env.multi_agent_env.MultiAgentEnv.__init__(self)
        self.config = config = (config or {})
        self.stubborn_config: StubbornConfig = self.config.setdefault('stubborn_config',
                                                                      StubbornConfig())

        self.agents = tuple(self.stubborn_config.policy_by_agent)
        self.policies = tuple(self.stubborn_config.policy_by_agent.values())

        self.mean_handicap = config.get('mean_handicap', None)
        self.handicap_difference = config.get('handicap_difference', None)

        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=int,
        )
        self.observation_space = gym.spaces.Dict({
            'i_turn': gym.spaces.Box(
                low=0,
                high=self.stubborn_config.episode_length,
                shape=(1,),
                dtype=int,
            ),
            'mean_completed_skirmish_length': gym.spaces.Box(
                low=0,
                high=self.stubborn_config.episode_length,
                shape=(1,),
                dtype=np.float32,
            ),
            'handicap': gym.spaces.Box(
                low=self.stubborn_config.min_handicap,
                high=self.stubborn_config.max_handicap,
                shape=(1,),
                dtype=np.float32,
            ),
            'reward_estimates': gym.spaces.Box(
                low=self.stubborn_config.min_reward_observation,
                high=self.stubborn_config.max_reward_observation,
                shape=(2,),
                dtype=np.float32,
            ),
            'biggest_reward_according_to_estimates': gym.spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=int,
            ),
            'agreeabilities': gym.spaces.Box(
                low=0,
                high=1,
                shape=(2,),
                dtype=np.float32
            ),
            'i_skurn_capped': gym.spaces.Box(
                low=0,
                high=self.stubborn_config.i_turn_relative_to_current_skirmish_observation_cap,
                shape=(1,),
                dtype=int,
            ),
            'we_played_left_reward_this_skirmish': gym.spaces.Box(
                low=False,
                high=True,
                shape=(1,),
                dtype=bool,
            ),
            'we_played_right_reward_this_skirmish': gym.spaces.Box(
                low=False,
                high=True,
                shape=(1,),
                dtype=bool,
            ),
            'zebras': gym.spaces.Box(
                low=0,
                high=1,
                shape=(2,),
                dtype=np.float32
            ),
            'yankee': gym.spaces.Box(
                low=-self.stubborn_config.max_reward,
                high=self.stubborn_config.max_reward,
                shape=(1,),
                dtype=np.float32
            ),
        })

        self._agent_ids = set(self.agents)
        self.reset()


    def make_initial_state(self) -> StubbornState:
        return StubbornState.make_initial(stubborn_config=self.stubborn_config)

    @staticmethod
    def sample_episode_to_text(algorithm: ray.rllib.algorithms.Algorithm) -> str:
        states = tuple(StubbornEnv.sample_episode(algorithm))
        return '\n'.join(state.text for state in states)
