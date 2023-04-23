# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

from typing import Mapping, Any, Container
import io
import time
import pathlib
import hashlib
import itertools
import re
import copy

import numpy as np
import ray.rllib.env.multi_agent_env

from stubborn.county import misc


def normalize_weights(weights: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    first_key = sorted(weights)[0]
    if (worker_snippet_match := re.search('_wk[0-9]+(?=/)', first_key)):
        fixed_weights = {}
        worker_snippet = worker_snippet_match.group(0)
        for key, value in weights.items():
            assert key.count(worker_snippet) == 1
            fixed_key = key.replace(worker_snippet, '')
            assert len(key) - len(fixed_key) == len(worker_snippet)
            fixed_weights[fixed_key] = value.copy()
        return fixed_weights
    else:
        return copy.deepcopy(weights)



class PolicySnapshot:
    def __init__(self, policy_or_weights: Mapping[str, np.ndarray] | ray.rllib.Policy) -> None:
        self.weights = normalize_weights(
            policy_or_weights.get_weights()
            if isinstance(policy_or_weights, ray.rllib.Policy)
            else policy_or_weights
        )
        self.hash = hashlib.sha512(
            b''.join(
                array.tobytes() for array in tuple(zip(*sorted(self.weights.items())))[1]
            )
        ).hexdigest()[:6]


    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PolicySnapshot):
            return (
                (self.weights.keys() == other.weights.keys()) and
                all(np.array_equal(value, other.weights[key])
                    for key, value in self.weights.items())
            )
        else:
            return NotImplemented

    def __hash__(self) -> str:
        return self.hash

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self.hash}>'


    def export_to_algorithm(self,
                            algorithm: ray.rllib.algorithms.Algorithm,
                            policy_name: str
                            ) -> None:
        CultureSnapshot({policy_name: self,}).export_to_algorithm(algorithm)

    def compickle_to_bytes(self) -> bytes:
        return misc.compickle_to_bytes(self.weights)


    @staticmethod
    def uncompickle_from_bytes(compickled_bytes: bytes) -> PolicySnapshot:
        return PolicySnapshot(misc.uncompickle_from_bytes(compickled_bytes))

    def __neg__(self) -> PolicySnapshot:
        return PolicySnapshot({key: -value for key, value in self.weights.items()})



class CultureSnapshot(Mapping):
    def __init__(self, policy_snapshot_by_policy_name: Mapping[str, PolicySnapshot]) -> None:
        self.policy_snapshot_by_policy_name = dict(policy_snapshot_by_policy_name)

    __getitem__ = lambda self, key: self.policy_snapshot_by_policy_name[key]
    __len__ = lambda self: len(self.policy_snapshot_by_policy_name)
    __iter__ = lambda self: iter(self.policy_snapshot_by_policy_name)
    __repr__ = lambda self: f'{type(self).__name__}({self.policy_snapshot_by_policy_name})'


    @staticmethod
    def import_from_algorithm(algorithm: ray.rllib.algorithms.Algorithm, *,
                              exclude_policy_names: Container[str] = ()) -> CultureSnapshot:
        return CultureSnapshot(
            {
                policy_name: PolicySnapshot(policy)
                for policy_name, policy in
                algorithm.workers.local_worker().policy_map.items()
                if policy_name not in exclude_policy_names
            }
        )


    def export_to_algorithm(self,
                            algorithm: ray.rllib.algorithms.Algorithm,
                            ) -> None:
        for policy_name, policy_snapshot in self.items():
            algorithm.get_policy(policy_name).set_weights(policy_snapshot.weights)
        algorithm.workers.sync_weights(self)
        for i in range(10):
            time.sleep(i)
            remote_workers = algorithm.workers.remote_workers()
            for worker, (policy_name, policy_snapshot) in itertools.product(remote_workers,
                                                                            self.items()):
                human_policy_weights_on_worker = ray.get(
                                                worker.get_weights.remote(policy_name))[policy_name]
                if PolicySnapshot(human_policy_weights_on_worker) != policy_snapshot:
                    continue

            # Confirmed all policies were propogated to remote workers.
            return

        raise RuntimeError("Policies didn't propogate to all remote workers.")

    def save(self, file_or_path: pathlib.Path | io.BufferedWriter) -> None:
        misc.compickle(self, file_or_path)

    @staticmethod
    def load(file_or_path: pathlib.Path | io.BufferedWriter) -> CultureSnapshot:
        result = misc.uncompickle(file_or_path)
        if not isinstance(result, CultureSnapshot):
            raise ValueError
        return result

    def __neg__(self) -> CultureSnapshot:
        return CultureSnapshot(
            {policy_name: - policy_snapshot for policy_name, policy_snapshot in self.items()}
        )
