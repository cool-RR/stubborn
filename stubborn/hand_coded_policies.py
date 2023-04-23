# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import ray.rllib.env.multi_agent_env
import ray.rllib.algorithms.callbacks
from ray.rllib.utils.typing import ModelWeights
import ray.rllib.algorithms.a2c
import ray.rllib.algorithms.a3c
import ray.rllib.algorithms.ppo
import ray.rllib.algorithms.sac
import ray.rllib.algorithms.appo
import ray.rllib.algorithms.impala
import ray.rllib.algorithms.maddpg
import numpy as np


class HandCodedHumanPolicy(ray.rllib.policy.policy.Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space_for_sampling = self.action_space

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        # if self.config.get("ignore_action_bounds", False) and isinstance(
            # self.action_space, gym.spaces.Box
        # ):
            # self.action_space_for_sampling = gym.spaces.Box(
                # -float("inf"),
                # float("inf"),
                # shape=self.action_space.shape,
                # dtype=self.action_space.dtype,
            # )
        # else:
            # self.action_space_for_sampling = self.action_space

    def init_view_requirements(self):
        from ray.rllib.policy.sample_batch import SampleBatch
        super().init_view_requirements()
        # Disable for_training and action attributes for SampleBatch.INFOS column
        # since it can not be properly batched.
        vr = self.view_requirements[SampleBatch.INFOS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        **kwargs
    ):
        actions = []
        raise NotImplementedError('This should be converted to use Move')
        for observation in obs_batch:
            *_, left_reward_estimate, right_reward_estimate = observation
            actions.append(
                np.array(
                    [0 if left_reward_estimate >= right_reward_estimate else 1],
                    dtype=int
                )
                # 0 if left_reward_estimate >= right_reward_estimate else 1
            )
        # raise ZeroDivisionError(repr(self.action_space_for_sampling))
        # for action in actions:
            # assert action in self.action_space_for_sampling
        return actions, [], {}

    def compute_single_action(self, *args, **kwargs):
        raw_action, *rest = ray.rllib.policy.policy.Policy.compute_single_action(self, *args,
                                                                                 **kwargs)
        assert isinstance(raw_action, list)
        return np.array(raw_action, dtype=np.float32), *rest


    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        raise NotImplementedError
        # return np.array([random.random()] * len(obs_batch))

    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        from ray.rllib.policy.sample_batch import SampleBatch
        import tree
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(
                    lambda s: s[None], self.observation_space.sample()
                ),
            }
        )


class HandCodedRobotPolicy(ray.rllib.policy.policy.Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space_for_sampling = self.action_space

        # Whether for compute_actions, the bounds given in action_space
        # should be ignored (default: False). This is to test action-clipping
        # and any Env's reaction to bounds breaches.
        # if self.config.get("ignore_action_bounds", False) and isinstance(
            # self.action_space, gym.spaces.Box
        # ):
            # self.action_space_for_sampling = gym.spaces.Box(
                # -float("inf"),
                # float("inf"),
                # shape=self.action_space.shape,
                # dtype=self.action_space.dtype,
            # )
        # else:
            # self.action_space_for_sampling = self.action_space

    def init_view_requirements(self):
        from ray.rllib.policy.sample_batch import SampleBatch
        super().init_view_requirements()
        # Disable for_training and action attributes for SampleBatch.INFOS column
        # since it can not be properly batched.
        vr = self.view_requirements[SampleBatch.INFOS]
        vr.used_for_training = False
        vr.used_for_compute_actions = False

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        **kwargs
    ):
        actions = []

        for observation in obs_batch:
            handicap, last_human_move, *_, left_reward_estimate, right_reward_estimate = observation
            # assert last_human_move in (-1, 1)
            raise NotImplementedError('This should be converted to use Move')
            actions.append(
                np.array(
                    # [random.choice((0, 1))],
                    [0 if last_human_move in (0, -1) else 1],
                    dtype=int
                )
                # 0 if left_reward_estimate >= right_reward_estimate else 1
            )
        # raise ZeroDivisionError(repr(self.action_space_for_sampling))
        # for action in actions:
            # assert action in self.action_space_for_sampling
        return actions, [], {}

    def compute_single_action(self, *args, **kwargs):
        raw_action, *rest = ray.rllib.policy.policy.Policy.compute_single_action(self, *args,
                                                                                 **kwargs)
        assert isinstance(raw_action, list)
        return np.array(raw_action, dtype=np.float32), *rest


    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    def compute_log_likelihoods(
        self,
        actions,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
    ):
        raise NotImplementedError
        # return np.array([random.random()] * len(obs_batch))

    def get_weights(self) -> ModelWeights:
        """No weights to save."""
        return {}

    def set_weights(self, weights: ModelWeights) -> None:
        """No weights to set."""
        pass

    def _get_dummy_batch_from_view_requirements(self, batch_size: int = 1):
        from ray.rllib.policy.sample_batch import SampleBatch
        import tree
        return SampleBatch(
            {
                SampleBatch.OBS: tree.map_structure(
                    lambda s: s[None], self.observation_space.sample()
                ),
            }
        )

