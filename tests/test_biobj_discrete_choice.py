"""Tests for the biobj discrete choice algorithm."""

import pytest

from paretochat.domain.problem import ScalarizationKind
from paretochat.domain.scoring import ObjectiveBound, ObjectiveParameter, ObjectiveParameterKind, ObjectiveScalarization
from paretochat.examples.biobj_discrete_choice.main import (
    BiObjDiscreteChoiceAlgorithm,
    ZeroOneScaler,
    ZeroOneScalingFunction,
)

# --- ZeroOneScaler ---


def test_zero_one_scaler_maps_bounds():
    scaler = ZeroOneScaler(min_value=10.0, max_value=50.0)
    assert scaler.scale(10.0) == pytest.approx(0.0)
    assert scaler.scale(50.0) == pytest.approx(1.0)
    assert scaler.scale(30.0) == pytest.approx(0.5)


def test_zero_one_scaler_degenerate_range():
    scaler = ZeroOneScaler(min_value=5.0, max_value=5.0)
    assert scaler.scale(5.0) == 0.0


def test_zero_one_scaler_roundtrip():
    scaler = ZeroOneScaler(min_value=100.0, max_value=500.0)
    original = 250.0
    assert scaler.descale(scaler.scale(original)) == pytest.approx(original)


# --- ZeroOneScalingFunction ---


def test_zero_one_scaling_function_transforms_all_objectives():
    scalers = {
        "delivery_time": ZeroOneScaler(30.0, 50.0),
        "cost": ZeroOneScaler(5000.0, 100000.0),
    }
    sf = ZeroOneScalingFunction(scalers)

    scaled = sf.transform({"delivery_time": 40.0, "cost": 52500.0})

    assert scaled["delivery_time"] == pytest.approx(0.5)
    assert scaled["cost"] == pytest.approx(0.5)


def test_zero_one_scaling_function_inverse_roundtrip():
    scalers = {
        "delivery_time": ZeroOneScaler(30.0, 50.0),
        "cost": ZeroOneScaler(5000.0, 100000.0),
    }
    sf = ZeroOneScalingFunction(scalers)

    original = {"delivery_time": 38.0, "cost": 9000.0}
    assert sf.inverse_transform(sf.transform(original)) == pytest.approx(original)


# --- BiObjDiscreteChoiceAlgorithm ---


@pytest.fixture(scope="module")
def algorithm():
    return BiObjDiscreteChoiceAlgorithm()


def _weighted_sum(delivery_time_weight: float, cost_weight: float) -> ObjectiveScalarization:
    return ObjectiveScalarization(
        kind=ScalarizationKind.WEIGHTED_SUM,
        parameters=[
            ObjectiveParameter(
                objective_name="delivery_time", kind=ObjectiveParameterKind.WEIGHT, value=delivery_time_weight
            ),
            ObjectiveParameter(objective_name="cost", kind=ObjectiveParameterKind.WEIGHT, value=cost_weight),
        ],
    )


def test_run_cost_heavy_weight_picks_cheap_solution(algorithm):
    """With weight almost entirely on cost, the lowest-scoring normalised solution wins.

    Computed manually from solutions.csv with dt_weight=0.01, cost_weight=0.99 and
    ZeroOneScaler ranges dt∈[30,50], cost∈[5000,101000]:
      dt=39 → scaled 0.45,  cost=5000.5 → scaled 5.2e-6  →  score ≈ 0.00451  ← winner
      dt=40 → scaled 0.50,  cost=5000.4 → scaled 4.2e-6  →  score ≈ 0.00500
    The next cheapest-cost solutions all have higher delivery_time, raising their score above the winner.
    """
    result = algorithm.run(_weighted_sum(delivery_time_weight=0.01, cost_weight=0.99))

    assert result is not None
    assert result.objective_values["delivery_time"] == pytest.approx(39.0)
    assert result.objective_values["cost"] == pytest.approx(5000.5)


def test_run_delivery_time_heavy_weight_picks_fastest_solution(algorithm):
    """With weight almost entirely on delivery_time, the returned solution should have the minimum delivery_time."""
    result = algorithm.run(_weighted_sum(delivery_time_weight=0.99, cost_weight=0.01))

    assert result is not None
    min_delivery_time = min(s.objective_values["delivery_time"] for s in algorithm._solutions)
    assert result.objective_values["delivery_time"] == pytest.approx(min_delivery_time)


def test_run_bound_filters_solutions(algorithm):
    """A max_delivery_time bound excludes solutions above the threshold."""
    scalarization = ObjectiveScalarization(
        kind=ScalarizationKind.WEIGHTED_SUM,
        parameters=[
            ObjectiveParameter(objective_name="delivery_time", kind=ObjectiveParameterKind.WEIGHT, value=0.5),
            ObjectiveParameter(objective_name="cost", kind=ObjectiveParameterKind.WEIGHT, value=0.5),
        ],
        bounds=[ObjectiveBound(objective_name="delivery_time", max_value=35.0)],
    )

    result = algorithm.run(scalarization)

    assert result is not None
    assert result.objective_values["delivery_time"] <= 35.0


def test_run_returns_none_when_no_solutions_pass_bounds(algorithm):
    """A bound that no solution satisfies causes run() to return None."""
    scalarization = ObjectiveScalarization(
        kind=ScalarizationKind.WEIGHTED_SUM,
        bounds=[ObjectiveBound(objective_name="delivery_time", max_value=1.0)],
    )

    result = algorithm.run(scalarization)

    assert result is None
