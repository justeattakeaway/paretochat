"""Base class for algorithms."""

from abc import ABC, abstractmethod

from paretochat.domain.problem import AlgorithmInfo, Solution
from paretochat.domain.scoring import ObjectiveScalarization


class Algorithm(ABC):
    """A class representing an algorithm.

    The use cases should use this base class to wrap the actual algorithm implementation.
    """

    @property
    @abstractmethod
    def algorithm_info(self) -> AlgorithmInfo:
        """Return the algorithm info."""
        raise NotImplementedError

    @abstractmethod
    def run(self, objective_scalarization: ObjectiveScalarization) -> Solution | None:
        """Run the algorithm with the given objective scalarization."""
        raise NotImplementedError
