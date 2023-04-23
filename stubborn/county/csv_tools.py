# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import csv
import pathlib
from typing import Mapping, Any, Optional, ContextManager, Iterable, Type
from types import TracebackType as Traceback


class CSVWriter(ContextManager):
    def __init__(self, path: str | pathlib.Path, fields: Iterable[str], *,
                 overwrite: bool = True) -> None:
        self.path = pathlib.Path(path)
        self.fields = tuple(fields)
        self.overwrite = overwrite

    def __enter__(self) -> CSVWriter:
        self._file = self.path.open('w' if self.overwrite else 'a', newline='')
        self._dict_writer = csv.DictWriter(self._file, fieldnames=self.fields)
        if self.overwrite:
            self._dict_writer.writeheader()
            self._file.flush()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException],
                 traceback: Optional[Traceback]) -> Optional[bool]:
        try:
            self._dict_writer = None
        finally:
            self._file.close()
            self._file = None

    def write_row(self, row: Mapping[str, Any], **row_extra: Any) -> None:
        self._dict_writer.writerow(dict(row) | row_extra)
        self._file.flush()

    def write_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        self._dict_writer.writerows(rows)
        self._file.flush()

