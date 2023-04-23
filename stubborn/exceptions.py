# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations


class StubbornException(Exception):
    pass


class SkirmishNotFinishedError(StubbornException):
    pass


class NoFinishedSkirmishesError(StubbornException):
    pass