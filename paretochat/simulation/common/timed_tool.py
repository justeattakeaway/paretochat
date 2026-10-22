from time import perf_counter
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field


class TimedTool(BaseTool):
    """Base tool that measures synchronous execution durations."""

    execution_count: int = 0
    total_execution_time_seconds: float = 0.0
    execution_durations_seconds: list[float] = Field(default_factory=list)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        start_time = perf_counter()
        try:
            return self._run_implementation(*args, **kwargs)
        finally:
            elapsed_seconds = perf_counter() - start_time
            self.execution_count += 1
            self.total_execution_time_seconds += elapsed_seconds
            self.execution_durations_seconds.append(elapsed_seconds)

    def _run_implementation(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses must implement _run_implementation.")
