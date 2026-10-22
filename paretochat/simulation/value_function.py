from abc import ABC, abstractmethod
import random
from typing import Dict, List

from paretochat.domain.problem import ObjectiveSense
from paretochat.simulation.base import ValueFunctionType


class ValueFunction(ABC):
    """Abstract base class for a value function."""

    @abstractmethod
    def evaluate(self, objective_values: Dict[str, float]) -> float:
        """Evaluate the value of a solution based on its objective values."""
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        """Return the type of the value function."""
        pass


class LinearValueFunction(ValueFunction):
    """Linear value function: w1*o1 + w2*o2 + ..."""

    def __init__(self, weights: Dict[str, float], maximize: bool = True):
        self.weights = weights
        self.maximize = maximize

    def evaluate(self, objective_values: Dict[str, float]) -> float:
        value = 0.0
        for obj_name, weight in self.weights.items():
            if obj_name not in objective_values:
                raise ValueError(f"Objective values are missing value for objective '{obj_name}'.")
            value += weight * objective_values[obj_name]
        return value if self.maximize else -value

    @property
    def type(self) -> str:
        return ValueFunctionType.LINEAR.value


class NonLinearValueFunction(ValueFunction):
    """
    Non-linear value function (Cobb-Douglas utility):
    U = o1^w1 * o2^w2 * ...
    Assumes objective values are positive.
    """

    def __init__(self, weights: Dict[str, float], maximize: bool = True):
        self.weights = weights
        self.maximize = maximize

    def evaluate(self, objective_values: Dict[str, float]) -> float:
        value = 1.0
        for obj_name, weight in self.weights.items():
            if obj_name not in objective_values:
                raise ValueError(f"Objective values are missing value for objective '{obj_name}'.")
            val = objective_values[obj_name]
            # Cobb-Douglas requires positive values.
            if val < 0:
                raise ValueError(f"Objective value for '{obj_name}' must be non-negative for Cobb-Douglas utility.")
            # We use a small epsilon for 0 values to avoid zeroing out everything if one obj is 0.
            val = max(val, 1e-6)
            value *= val**weight
        return value if self.maximize else -value

    @property
    def type(self) -> str:
        return ValueFunctionType.NONLINEAR.value


class WeigtedSquaredSumsDistanceToIdealPoint(ValueFunction):
    """Negative weighted sum of squared distances to an ideal objective vector."""

    def __init__(self, weights: Dict[str, float], ideal_point: Dict[str, float]):
        self.weights = weights
        self.ideal_point = ideal_point

    def evaluate(self, objective_values: Dict[str, float]) -> float:
        weighted_squared_sum_distance = 0.0
        for obj_name, weight in self.weights.items():
            if obj_name not in self.ideal_point:
                raise ValueError(f"Ideal point is missing value for objective '{obj_name}'.")
            if obj_name not in objective_values:
                raise ValueError(f"Objective values are missing value for objective '{obj_name}'.")
            ideal_value = self.ideal_point[obj_name]
            distance = objective_values[obj_name] - ideal_value
            weighted_squared_sum_distance += weight * (distance**2)
        return -weighted_squared_sum_distance

    @property
    def type(self) -> str:
        return ValueFunctionType.SUM_SQUARED_DISTANCE.value


class ValueFunctionFactory:
    """Factory to create value functions."""

    @staticmethod
    def _create_random_normalized_weights(objective_names: List[str], seed: int = None) -> Dict[str, float]:
        if seed is not None:
            random.seed(seed)

        raw_weights = [random.random() for _ in range(len(objective_names))]
        total_weight = sum(raw_weights)
        if total_weight == 0:
            total_weight = 1
            raw_weights = [1.0 / len(objective_names)] * len(objective_names)

        return {name: float(w) / total_weight for name, w in zip(objective_names, raw_weights, strict=True)}

    @staticmethod
    def create_random_linear(
        objective_names: List[str], maximize: bool = True, seed: int = None
    ) -> LinearValueFunction:
        weights = ValueFunctionFactory._create_random_normalized_weights(objective_names, seed=seed)
        return LinearValueFunction(weights, maximize=maximize)

    @staticmethod
    def create_random_nonlinear(
        objective_names: List[str], maximize: bool = True, seed: int = None
    ) -> NonLinearValueFunction:
        weights = ValueFunctionFactory._create_random_normalized_weights(objective_names, seed=seed)
        return NonLinearValueFunction(weights, maximize=maximize)

    @staticmethod
    def create_random_weigted_squared_sums_distance_to_ideal_point(
        objective_names: List[str],
        objective_senses: Dict[str, ObjectiveSense],
        pareto_front: List[Dict[str, float]],
        seed: int = None,
    ) -> WeigtedSquaredSumsDistanceToIdealPoint:
        if not pareto_front:
            raise ValueError("Cannot derive ideal point from an empty Pareto front.")

        ideal_point: Dict[str, float] = {}
        for objective_name in objective_names:
            objective_values = [point[objective_name] for point in pareto_front]
            sense = objective_senses[objective_name]
            if sense == ObjectiveSense.MAXIMIZE:
                ideal_point[objective_name] = max(objective_values)
            elif sense == ObjectiveSense.MINIMIZE:
                ideal_point[objective_name] = min(objective_values)
            else:
                raise ValueError(f"Unknown objective sense for {objective_name}: {sense}")

        weights = ValueFunctionFactory._create_random_normalized_weights(objective_names, seed=seed)
        return WeigtedSquaredSumsDistanceToIdealPoint(weights=weights, ideal_point=ideal_point)
