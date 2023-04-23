# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import logging

from . import filtros
filtros.activate()

from .base_env import BaseEnv
from .base_state import BaseState
from . import constants

def init_ray():
    import ray
    if not ray._private.worker.global_worker.connected:
        ray.init(logging_level=logging.WARN)
