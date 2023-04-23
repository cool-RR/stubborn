# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

# pytype: disable=module-attr
from __future__ import annotations

from typing import Optional, Mapping, Iterable, Callable
import itertools
import abc
import ray.rllib.env.multi_agent_env

from .base_state import BaseState
from stubborn.county.constants import ALL_AGENTS
from stubborn.county.typing import Agent, Action, Observation
from stubborn.county.misc import compute_actions_for_all_agents


class BaseEnv(ray.rllib.env.multi_agent_env.MultiAgentEnv, abc.ABC):

    def __init__(self, config: Optional[Mapping] = None) -> None:
        ray.rllib.env.multi_agent_env.MultiAgentEnv.__init__(self)
        self.states = []

    @abc.abstractmethod
    def make_initial_state(self) -> BaseState:
        raise NotImplementedError

    def reset(self,
              initial_states: None | BaseState | Iterable[BaseState] = None,
              /) -> Mapping[Agent, Observation]:
        self.states = (
            [self.make_initial_state()] if initial_states is None else
            [initial_states] if isinstance(initial_states, BaseState) else
            list(initial_states)
        )
        return self.state.observation_by_agent

    @property
    def state(self) -> BaseState:
        return self.states[-1]

    @property
    def observation_by_agent(self) -> Mapping[Agent, Observation]:
        return self.states[-1].observation_by_agent

    def step(self, actions: Mapping[Agent, Action]) -> tuple[Mapping, Mapping, Mapping, Mapping]:
        self.states.append(self.state.step(actions))
        return (self.state.observation_by_agent, self.state.reward_by_agent,
                self.state.done_by_agent, {})

    def render(self, mode: Optional[str] = None) -> str:
        return self.state.text

    def play(self,
             algorithm: ray.rllib.algorithms.Algorithm,
             n: Optional[int] = None,
             stop_condition: Optional[Callable[BaseState, bool]] = None,
             ) -> Iterator[BaseState]:
        for i in (range(n) if n else itertools.count()):
            if stop_condition is not None and stop_condition(self.state):
                return
            actions = compute_actions_for_all_agents(algorithm, self)
            observation_by_agent, reward_by_agent, done_by_agent, infos = self.step(actions)
            yield self.state
            if done_by_agent[ALL_AGENTS]:
                return

    @classmethod
    def sample_episode(cls,
                       algorithm: ray.rllib.algorithms.Algorithm,
                       n: Optional[int] = None,
                       stop_condition: Optional[Callable[BaseState, bool]] = None,
                       initial_states: None | BaseState | Iterable[BaseState] = None,
                       ) -> Iterator[BaseState]:
        env = cls(config=algorithm.config['env_config'])
        env.reset(initial_states)
        yield env.state
        new_n = None if (n is None) else (n - 1)
        yield from env.play(algorithm, n=new_n, stop_condition=stop_condition)


