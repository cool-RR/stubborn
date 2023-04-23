# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import numbers
from typing import Any
from ray.rllib.utils.typing import AgentID, EnvActionType as Action, PolicyID, MultiAgentDict

RealNumber = float | int | numbers.Real

Agent = str
AgentOrAll = str
Observation = Any
Reward = RealNumber

del Any, numbers