"""Base class for scaling functions."""

from abc import ABC, abstractmethod


class ScalingFunction(ABC):
    """Scale the objective values into the same range."""

    @abstractmethod
    def fit(self):
        """Fit the scaling function to the data."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, objective_values: dict[str, float]) -> dict[str, float]:
        """Transform the objective values into the same range."""
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, objective_values: dict[str, float]) -> dict[str, float]:
        """Inverse transform the objective values into the original range."""
        raise NotImplementedError
