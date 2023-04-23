# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import math
import random
import re
import copy
from typing import Iterable

import numpy as np

import ray.rllib.env.multi_agent_env

from stubborn.county.typing import Agent, RealNumber
from .stubborn_state import StubbornState


def triangle_lengths_to_sides(a_length: RealNumber, b_length: RealNumber, c_length: RealNumber
                              ) -> tuple[RealNumber, RealNumber, RealNumber]:
    return (
        math.acos((b_length ** 2 + c_length ** 2 - a_length ** 2) /
                  (2 * b_length * c_length)),
        math.acos((a_length ** 2 + c_length ** 2 - b_length ** 2) /
                  (2 * a_length * c_length)),
        math.acos((a_length ** 2 + b_length ** 2 - c_length ** 2) /
                  (2 * a_length * b_length))
    )

def are_weights_equal(local_weights: dict[str, np.ndarray],
                      worker_weights: dict[str, np.ndarray]) -> bool:
    fixed_worker_weights = copy.deepcopy(worker_weights)
    first_worker_key = sorted(fixed_worker_weights)[0]
    worker_snippet = re.search('_wk[0-9]+(?=/)', first_worker_key).group(0)
    for key, value in tuple(fixed_worker_weights.items()):
        fixed_key = key.replace(worker_snippet, '')
        assert len(key) - len(fixed_key)  == len(worker_snippet)
        fixed_worker_weights[fixed_key] = value
        del fixed_worker_weights[key]

    return (
        set(local_weights) == set(fixed_worker_weights) and
        all(np.array_equal(value, fixed_worker_weights[key])
            for key, value in local_weights.items())
    )


def clamp(number: RealNumber, low: RealNumber, high: RealNumber) -> RealNumber:
    if number < low:
        return low
    elif number > high:
        return high
    else:
        return number


def shuffled(iterable: Iterable[Any]) -> tuple[Any, ...]:
    list_ = list(iterable)
    random.shuffle(list_)
    return tuple(list_)


def get_move_by_state(algorithm: ray.rllib.algorithms.Algorithm,
                      states: Iterable[StubbornState],
                      agent: Agent) -> dict[StubbornState, int]:
    from .stubborn_state import Move
    states = tuple(states)
    policy = algorithm.config['multiagent']['policy_mapping_fn'](agent)
    robot_policy = algorithm.get_policy(policy)
    observation_preprocessor = algorithm.workers.local_worker().preprocessors[policy]
    state_to_flat_observation = (
        lambda stubborn_state: observation_preprocessor.transform(
            stubborn_state.observation_by_agent[agent]
        )
    )
    actions, _, _ = robot_policy.compute_actions(
        np.stack(tuple(map(state_to_flat_observation, states))),
        explore=False
    )
    return dict(zip(states, map(Move.from_neural, actions), strict=True))


