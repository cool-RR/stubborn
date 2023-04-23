# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import threading
import pathlib
import time
import traceback as traceback_module
import sys
import datetime

class ThreadyKrueger(threading.Thread):
    _stop = False
    def __init__(self, path: pathlib.Path | str, *, start: bool = False,
                 interval_seconds: int = 60) -> None:
        self.path = pathlib.Path(path)
        self.interval_seconds = interval_seconds
        if self.path.exists():
            assert self.path.is_dir()
        threading.Thread.__init__(self, daemon=True)
        if start:
            self.start()

    def run(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        while not self._stop:
            with (self.path / datetime.datetime.now().isoformat()).open('w') as f:
                for thread_id, frame in sys._current_frames().items():
                    f.write(f'Thread ID: {thread_id}\n')
                    f.write(f'{"".join(traceback_module.format_stack(frame))}\n')
            time.sleep(self.interval_seconds)

    def __del__(self) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop = True



