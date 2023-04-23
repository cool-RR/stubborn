# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

from typing import Optional
import statistics
import functools
import functools

import ray.rllib.env.multi_agent_env
import ray.rllib.algorithms.callbacks
from ray.rllib.utils.typing import PolicyID
import ray.rllib.algorithms.a2c
import ray.rllib.algorithms.a3c
import ray.rllib.algorithms.ppo
import ray.rllib.algorithms.sac
import ray.rllib.algorithms.appo
import ray.rllib.algorithms.impala
import ray.rllib.algorithms.maddpg
from ray import rllib

from stubborn.county.typing import RealNumber, PolicyID
from .stubborn_state import StubbornState, Move, RewardPack
from .stubborn_config import StubbornConfig


@functools.cache
def make_state_by_difference_and_length(stubborn_config: StubbornConfig, *,
                                        difference: RealNumber, length: int) -> StubbornState:
    return StubbornState(
        stubborn_config=stubborn_config,
        i_turn=length,
        reward_pack_by_i_first_turn={
            0: RewardPack(
                left_reward=(left_reward := stubborn_config.midpoint_reward + difference / 2),
                right_reward=(right_reward := stubborn_config.midpoint_reward - difference / 2),
                estimates_of_left_reward=(left_reward, left_reward),
                estimates_of_right_reward=(right_reward, right_reward),
                flips=(False, False),
            )
        },
        move_pairs=((Move.LEFT, Move.RIGHT),) * length,
        handicaps=(stubborn_config.min_handicap, stubborn_config.min_handicap),
        # Todo: should this be something else? constant?
        agreed_move_by_i_turn={},
    )

class Callbackino(ray.rllib.algorithms.callbacks.DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: ray.rllib.evaluation.RolloutWorker,
        base_env: ray.rllib.BaseEnv,
        policies: dict[PolicyID, ray.rllib.Policy],
        episode: ray.rllib.evaluation.Episode | ray.rllib.evaluation.episode_v2.EpisodeV2 |
            Exception,
        **kwargs,
        ) -> None:

        if isinstance(episode, Exception):
            return


    # @pysnooper.snoop(output='/home/ramrachum/Desktop/snoopy.txt', depth=1, max_variable_length=200,
                     # relative_time=True, color=False)
    def on_train_result(
        self,
        *,
        algorithm: ray.rllib.algorithms.Algorithm,
        result: dict,
        **kwargs,
        ) -> None:
        """Called at the end of Algorithm.train().

        Args:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """

        insistence_by_stubbornness = {}
        n_samples_per_stubbornness = 100
        stubbornnesses_to_check = (0, 1, 2, 3, 4)
        stubborn_config = algorithm.config['env_config']['stubborn_config']

        for stubbornness_to_check in stubbornnesses_to_check:
            insistence_samples = []
            for _ in range(n_samples_per_stubbornness):
                state = make_state_by_difference_and_length(stubborn_config,
                                                            difference=5,
                                                            length=stubbornness_to_check)
                move_neural = algorithm.compute_single_action(
                    state.observation_by_agent['agent_a'],
                    policy_id=stubborn_config.policy_by_agent['agent_a']
                )
                insistence_samples.append((move_neural[0] == Move.LEFT.to_neural()))
            insistence_by_stubbornness[stubbornness_to_check] = statistics.mean(
                map(float, insistence_samples)
            )


        result |= {
            'insistence_on_stubbornness_0': insistence_by_stubbornness[0],
            'insistence_on_stubbornness_1': insistence_by_stubbornness[1],
            'insistence_on_stubbornness_2': insistence_by_stubbornness[2],
            'insistence_on_stubbornness_3': insistence_by_stubbornness[3],
            'insistence_on_stubbornness_4': insistence_by_stubbornness[4],
        }

    def on_episode_end(
        self,
        *,
        worker: rllib.evaluation.RolloutWorker,
        base_env: rllib.env.BaseEnv,
        policies: dict[PolicyID, rllib.Policy],
        episode: Union[rllib.evaluation.Episode, rllib.evaluation.episode_v2.EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Runs when an episode is done.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
                In case of environment failures, episode may also be an Exception
                that gets thrown from the environment before the episode finishes.
                Users of this callback may then handle these error cases properly
                with their custom logics.
            env_index: The index of the sub-environment that ended the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        (env,) = base_env.get_sub_environments()
        last_state: StubbornState = env.state
        episode.custom_metrics |= {
            f'{agent}_agreeability': last_state.agreeabilities[i]
            for i, agent in enumerate(env.stubborn_config.policy_by_agent)
        }
        episode.custom_metrics |= {
            f'{agent}_zebra': last_state.zebras[i]
            for i, agent in enumerate(env.stubborn_config.policy_by_agent)
        }
        episode.custom_metrics |= {
            f'{agent}_yankee': last_state.yankees[i]
            for i, agent in enumerate(env.stubborn_config.policy_by_agent)
        }
        episode.custom_metrics |= {
            f'{agent}_glee_skurn_{j}': last_state.glee_by_skurn_pair[i][j]
            for i, agent in enumerate(env.stubborn_config.policy_by_agent)
            for j in range(3)
            if len(last_state.glee_by_skurn_pair[i]) >= j + 1
        }
        episode.custom_metrics |= {
            'mean_completed_skirmish_length': last_state.mean_completed_skirmish_length,
            'n_completed_skirmishes': len(last_state.completed_skirmish_lengths),
            'left_reward_popularity': last_state.left_reward_popularity,
            'right_reward_popularity': last_state.right_reward_popularity,
        }

