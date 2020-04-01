# Inspired by https://stackoverflow.com/a/30024601/854731
from contextlib import AbstractContextManager
from timeit import default_timer
from typing import Any, Optional


class ContextStopwatch(AbstractContextManager):
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @property
    def elapsed_time(self) -> float:
        assert self.start_time is not None
        assert self.end_time is not None

        return self.end_time - self.start_time

    def __enter__(self) -> AbstractContextManager:
        self.start_time = default_timer()
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.end_time = default_timer()
