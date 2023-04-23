# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

from typing import Optional, Any
import typing
import dataclasses
import random

import ray.rllib.env.multi_agent_env
import ray.rllib.algorithms.callbacks
from ray.rllib.utils.typing import PolicyID
import ray.tune
import ray.rllib.algorithms.ppo
import click

from stubborn.county import misc
from stubborn.county.typing import Agent, RealNumber
from . import defaults


@dataclasses.dataclass(kw_only=True)
class StubbornConfig:

    mean_handicap: RealNumber = defaults.DEFAULT_MEAN_HANDICAP
    handicap_difference: RealNumber = defaults.DEFAULT_HANDICAP_DIFFERENCE
    train_batch_size: int = defaults.DEFAULT_TRAIN_BATCH_SIZE
    learning_rate: RealNumber = defaults.DEFAULT_LEARNING_RATE
    n_generations: int = defaults.DEFAULT_N_GENERATIONS
    randomize_labels: bool = defaults.DEFAULTS_RANDOMIZE_LABELS

    min_reward: RealNumber = 0
    max_reward: RealNumber = 10
    episode_length: RealNumber = defaults.DEFAULT_EPISODE_LENGTH
    min_handicap: RealNumber = 1
    max_handicap: RealNumber = 10
    i_turn_relative_to_current_skirmish_observation_cap = 5
    # n_rounds_in_metrics = 4
    n_tune_samples = 15

    def __post_init__(self) -> None:

        self.midpoint_reward = (self.max_reward + self.min_reward) / 2
        self._reward_observation_cap_factor = 2
        self.min_reward_observation = (self.min_reward -
                                       self._reward_observation_cap_factor * self.max_handicap)
        self.max_reward_observation = (self.max_reward +
                                       self._reward_observation_cap_factor * self.max_handicap)


        self.policy_by_agent = {'agent_a': 'policy_a', 'agent_b': 'policy_b'}


    def policy_mapping_fn(self,
                          agent_id: Agent,
                          episode: Optional[ray.rllib.evaluation.episode.Episode] = None,
                          worker: Optional[ray.rllib.evaluation.rollout_worker.
                                           RolloutWorker] = None,
                          **kwargs) -> PolicyID:
        return self.policy_by_agent[agent_id]

    def get_nice_dict(self) -> dict[str, Any]:
        return {key: value for key, value in vars(self).items()
                if not key.startswith('_')}


    @classmethod
    def _get_field_type_for_click(cls, field_name: str) -> type:
        field_type_map = {
            RealNumber: float,
        }
        raw_field_type = typing.get_type_hints(StubbornConfig)[field_name]
        try:
            return field_type_map[raw_field_type]
        except KeyError:
            return raw_field_type


    @classmethod
    def add_options_to_click_command(cls, command: click.decorators.FC) -> click.decorators.FC:
        for field in cls.__dataclass_fields__.values():
            field: dataclasses.Field
            field_type = cls._get_field_type_for_click(field.name)
            dashed_field_name = field.name.replace('_', '-')
            option_kwargs = {
                'type': field_type,
                'default': (field.default,),
                'multiple': True,
                'show_default': True,
            }
            if field_type is bool:
                option_kwargs['is_flag'] = True
                name = f'--{dashed_field_name}/--not-{dashed_field_name}'
            else:
                name = f'--{dashed_field_name}'
            command = click.option(name, **option_kwargs)(command)
        return command

    def __hash__(self) -> int:
        return hash(
            (type(self),
             *(getattr(self, field) for field in self.__dataclass_fields__))
        )

    def _make_reward_estimates(self,
                               reward: RealNumber,
                               handicaps: tuple[RealNumber, RealNumber]
                               ) -> tuple[RealNumber, RealNumber]:
        return tuple(
            misc.clamp(random.gauss(reward, handicap),
                       self.min_reward_observation,
                       self.max_reward_observation) for handicap in handicaps
        )

    def _make_random_reward(self) -> RealNumber:
        # if (r := random.random()) < 0.9:
            # reward = random.uniform(0, 2)
        # elif r <= 0.95:
            # reward = random.uniform(2, 8)
        # else:
            # assert 0.95 <= r <= 1
            # reward = random.uniform(8, 10)
        reward = random.uniform(0, 10)
        assert self.min_reward <= reward <= self.max_reward
        return reward
