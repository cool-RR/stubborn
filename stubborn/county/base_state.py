# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

from typing import Mapping
import abc

from stubborn.county.typing import Agent, Action, AgentOrAll, Observation, Reward


class BaseState(abc.ABC):

    observation_by_agent: Mapping[Agent, Observation]

    reward_by_agent: Mapping[Agent, Reward]

    done_by_agent: Mapping[AgentOrAll, Reward]

    text: str

    @abc.abstractstaticmethod
    def make_initial() -> BaseState: # In the future: `-> Self:`
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, actions: Mapping[Agent, Action]) -> BaseState: # In the future: `-> Self:`
        raise NotImplementedError



