# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import functools
import pathlib
import math
import os
import datetime as datetime_module
import lzma
import random
import json
import pandas as pd
import sys
import subprocess
import itertools
import operator
import collections.abc
import contextlib
import pathlib
import pickle
import more_itertools
from typing import Any, Mapping, Optional, BinaryIO, Iterable, TextIO, Iterator
import io

import numpy as np
import ray.rllib

from .constants import STUBBORN_HOME
from stubborn.county import csv_tools
from .typing import Agent, Action, RealNumber
from stubborn import county
from . import filtros



def make_output_folder(big_folder_name: str) -> pathlib.Path:
    assert os.sep not in big_folder_name
    assert not big_folder_name.startswith('.')
    big_folder: pathlib.Path = STUBBORN_HOME / big_folder_name
    folder = big_folder / (datetime_module.datetime.now().isoformat()
                           .replace(':', '-').replace('.', '-').replace('T', '-'))
    folder.mkdir(parents=True)
    return folder


def parse_output_path(path_expression: str | pathlib.Path | None, big_folder_name: str,
                      *, file_name: Optional[str] = None) -> pathlib.Path:
    if path_expression is None:
        big_folder: pathlib.Path = STUBBORN_HOME / big_folder_name
        for folder in sorted(big_folder.iterdir(), reverse=True):
            if file_name is None:
                return folder
            elif (file_path := folder / file_name).exists():
                return file_path
        else:
            raise FileNotFoundError
    else:
        path = pathlib.Path(path_expression)
        if file_name is None:
            assert path.is_dir()
            return path
        assert file_name is not None
        if path.is_dir():
            file_path = path / file_name
        elif path.name != file_name:
            raise Exception
        else:
            file_path = path
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        return file_path


def sane_kwargs(function):
    @functools.wraps(function)
    def inner(config):
        return function(**config)
    return inner


def jsonl_to_dataframe(jsonl_path: str | pathlib.Path) -> pd.DataFrame:
    jsonl_path = pathlib.Path(jsonl_path)
    with jsonl_path.open() as file:
        entries = tuple(map(json.loads, file))
    return pd.DataFrame(entries)


def compute_actions_for_all_agents(algorithm: ray.rllib.algorithms.Algorithm,
                                   env: county.BaseEnv) -> dict[Agent, Action]:
    return {agent: algorithm.compute_single_action(
        env.observation_by_agent[agent],
        policy_id=algorithm.config['multiagent']['policy_mapping_fn'](agent)
    )
            for agent in env.get_agent_ids()}



def compickle(thing: Any, file_or_path: pathlib.Path | BinaryIO, /) -> None:
    with contextlib.ExitStack() as exit_stack:
        if isinstance(file_or_path, pathlib.Path):
            file_ = exit_stack.enter_context(file_or_path.open('wb'))
        else:
            file_ = file_or_path
        file_.write(lzma.compress(pickle.dumps(thing)))


def compickle_to_bytes(thing: Any)-> bytes:
    bytes_io = io.BytesIO()
    compickle(thing, bytes_io)
    return bytes_io.getvalue()

def uncompickle(file_or_path: pathlib.Path | BinaryIO, /) -> Any:
    with contextlib.ExitStack() as exit_stack:
        if isinstance(file_or_path, pathlib.Path):
            file_ = exit_stack.enter_context(file_or_path.open('rb'))
        else:
            file_ = file_or_path
        return pickle.loads(lzma.decompress(file_.read()))

def uncompickle_from_bytes(compickled_bytes: bytes) -> Any:
    return uncompickle(io.BytesIO(compickled_bytes))


class TroubleshootingDict(collections.abc.MutableMapping):
    def __init__(self, *args, **kwargs):
        self._map = dict(*args, **kwargs)

    def __getitem__(self, key):
        return self._map[key]

    def __delitem__(self, key):
        del self._map[key]

    def __setitem__(self, key, value):
        if isinstance(value, list):
            if all(isinstance(item, list) for item in value):
                value = TroubleshootingList(map(TroubleshootingList, value))
            else:
                value = TroubleshootingList(value)
        self._map[key] = value

    def __len__(self):
        return len(self._map)

    def __iter__(self):
        return iter(self._map)

    def __repr__(self):
        return f'{type(self).__name__}({self._map})'


class TroubleshootingList(collections.abc.MutableSequence):
    def __init__(self, *args, **kwargs):
        self._list = list(*args, **kwargs)

    def __getitem__(self, key):
        return self._list[key]

    def __delitem__(self, key):
        del self._list[key]

    def __setitem__(self, key, value):
        self._list[key] = value

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def insert(self, index, thing):
        self._list.insert(index, thing)

    def __repr__(self):
        return f'{type(self).__name__}({self._list})'



def all_equal(values: Iterable[Any]) -> bool:
    return all(itertools.starmap(operator.eq, more_itertools.windowed(values, 2)))


def cute_div(x: RealNumber, y: RealNumber, *, default: RealNumber = 0) -> RealNumber:
    try:
        return x / y
    except ZeroDivisionError:
        return default

# def cool_windowed(iterable: Iterable[Any], n: int = 2) -> Iterable[tuple[Any, Any]]:
    # sentinel = object()
    # for left, right in more_itertools.windowed(iterable, n, fillvalue=sentinel):
        # if sentinel in (left, right):
            # return
        # yield (left, right)

def clamp(number: RealNumber, low: RealNumber, high: RealNumber) -> RealNumber:
    if number < low:
        return low
    elif number > high:
        return high
    else:
        return number


class TeeStream:
    def __init__(self, original_stream: TextIO, path: pathlib.Path)-> None:
        self.original_stream = original_stream
        self.path = path

    def write(self, message: str) -> None:
        self.original_stream.write(message)
        with self.path.open('a') as file:
            file.write(message)

    def flush(self) -> None:
        self.original_stream.flush()

    def fileno(self) -> int:
        return self.original_stream.fileno()

    def close(self) -> None:
        pass


@contextlib.contextmanager
def tee_stdout(path: pathlib.Path) -> None:
    original_stdout = sys.stdout
    sys.stdout = TeeStream(original_stdout, path)
    try:
        yield
    finally:
        sys.stdout = original_stdout


@contextlib.contextmanager
def tee_stderr(path: pathlib.Path, *, ensure_filtros: bool = True) -> None:
    original_stderr = sys.stderr
    sys.stderr = TeeStream(original_stderr, path)
    try:
        if ensure_filtros:
            filtros.activate()
        yield
    finally:
        sys.stderr = original_stderr


class BaseRolloutReporter(collections.abc.Sequence):
    def __init__(self) -> None:
        self.rows = []

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.rows

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.rows[i]

    def __len__(self) -> int:
        return len(self.rows)

    def __reversed__(self) -> Iterator[dict[str, Any]]:
        yield from reversed(self.rows)

    def as_dataframe(self) -> pd.DataFrame:
        pd.DataFrame.from_dict(self.rows)

    def report(self, row_or_rows: Mapping[str, Any] | Iterable[Mapping[str, Any]], /) -> None:
        for row in self._parse_row_or_rows(row_or_rows):
            row = dict(row)
            self.rows.append(row)

    @staticmethod
    def _parse_row_or_rows(row_or_rows: Mapping[str, Any] | Iterable[Mapping[str, Any]]
                           ) -> Iterable[Mapping[str, Any]]:
        if isinstance(row_or_rows, collections.abc.Mapping):
            return (row_or_rows,)
        else:
            return row_or_rows

class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None  # Serialized as JSON null.
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)

class TuneRolloutReporter(BaseRolloutReporter):
    def report(self, row_or_rows: Mapping[str, Any] | Iterable[Mapping[str, Any]], /) -> None:
        import ray.tune
        for row in self._parse_row_or_rows(row_or_rows):
            BaseRolloutReporter.report(self, row)
            ray.tune.report(**row)

class BaseFileRolloutReporter(BaseRolloutReporter):
    def __init__(self, path: pathlib.Path) -> None:
        BaseRolloutReporter.__init__(self)
        self.path = path


class CsvRolloutReporter(BaseFileRolloutReporter):
    def report(self, row_or_rows: Mapping[str, Any] | Iterable[Mapping[str, Any]], /) -> None:
        for row in self._parse_row_or_rows(row_or_rows):
            BaseFileRolloutReporter.report(self, row)
            with csv_tools.CSVWriter(self.path, row.keys(),
                                     overwrite=(len(self.rows) == 1)) as csv_writer:
                csv_writer.write_row(row)

class JsonlRolloutReporter(BaseFileRolloutReporter):
    def report(self, row_or_rows: Mapping[str, Any] | Iterable[Mapping[str, Any]], /) -> None:
        for row in self._parse_row_or_rows(row_or_rows):
            BaseFileRolloutReporter.report(self, row)
            make_tree = lambda: collections.defaultdict(make_tree)
            tree = make_tree()
            for key, value in row.items():
                *parents, short_name = key.split('.')
                subtree = tree
                for parent in parents:
                    subtree = subtree[parent]
                subtree[short_name] = value

            with self.path.open('a') as file:
                json.dump(tree, file, cls=NumpyJsonEncoder)
                file.write('\n')



def shuffled(iterable: Iterable[Any]) -> tuple[Any, ...]:
    list_ = list(iterable)
    random.shuffle(list_)
    return tuple(list_)

def get_mean_dataframe_from_experiment_analysis(
    experiment_analysis: ray.tune.ExperimentAnalysis,
    fields_that_need_last: tuple[str] = ()) -> pd.DataFrame:

    rows = {
        path: df.mean(numeric_only=True).to_dict()
        for path, df in experiment_analysis.trial_dataframes.items()
    }

    for field_that_needs_last in fields_that_need_last:
        for path, df in experiment_analysis.trial_dataframes.items():
            rows[path][field_that_needs_last] = df[field_that_needs_last].values[-1]

    all_configs = experiment_analysis.get_all_configs(prefix=True)
    for path, config in all_configs.items():
        if path in rows:
            rows[path].update(config)
            rows[path].update(logdir=path)
    return pd.DataFrame(list(rows.values()))

def int_sqrt(x: RealNumber) -> int:
    sqrt_float = math.sqrt(x)
    sqrt_int = int(sqrt_float)
    if not sqrt_int == sqrt_float:
        raise ValueError(f"Got number {x} when expecting a number that's a square of an "
                         f"integer. Alas, the sqrt of {x} is a non-int: {sqrt_float}")
    return sqrt_int


def make_one_hot(i: int, n: int) -> np.ndarray:
    array = np.zeros((n,), dtype=np.int8)
    array[i] = 1
    return array
