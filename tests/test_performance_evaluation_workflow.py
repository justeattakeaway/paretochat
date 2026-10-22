"""Tests for ValueFunctionFactory + PerformanceEvaluator end-to-end workflows."""

import pytest

from paretochat.domain.problem import Solution
from paretochat.simulation.ai_dm import Decision, SimulationResult
from paretochat.simulation.evaluator import PerformanceEvaluator
from paretochat.simulation.simulation_problem import SimulationProblemInstance
from paretochat.simulation.value_function import ValueFunctionFactory


def _make_instance(pareto_front: list) -> SimulationProblemInstance:
    return SimulationProblemInstance(name="test", input_data=None, pareto_front=pareto_front)


def _wrap_solution(point: dict, num_eval: int = 5) -> SimulationResult:
    return SimulationResult(
        num_eval_sols=num_eval,
        final_decision=Decision(solution=Solution(objective_values=point), reasoning="test"),
    )


def test_optimal_solution_scores_rank_1(small_pareto_front):
    """The best Pareto-front point achieves rank 1 and zero value gap."""
    vf = ValueFunctionFactory.create_random_linear(["profit", "cost"], maximize=True, seed=42)
    best_point = max(small_pareto_front, key=vf.evaluate)

    metrics = PerformanceEvaluator(_make_instance(small_pareto_front)).evaluate(_wrap_solution(best_point), vf)

    assert metrics.solution_found
    assert metrics.is_optimal
    assert metrics.found_value_rank == 1
    assert metrics.found_value_rank_normalized == pytest.approx(1.0)
    assert metrics.value_gap == pytest.approx(0.0)
    assert metrics.pareto_size == 5


def test_suboptimal_solution_rank_and_gap(small_pareto_front):
    """The worst Pareto-front point has the highest rank and a positive value gap."""
    vf = ValueFunctionFactory.create_random_linear(["profit", "cost"], maximize=True, seed=42)
    worst_point = min(small_pareto_front, key=vf.evaluate)

    metrics = PerformanceEvaluator(_make_instance(small_pareto_front)).evaluate(_wrap_solution(worst_point), vf)

    assert metrics.solution_found
    assert not metrics.is_optimal
    assert metrics.found_value_rank > 1
    assert metrics.found_value_rank_normalized == pytest.approx(0.0)
    assert metrics.value_gap > 0.0


def test_no_decision_gives_solution_not_found(small_pareto_front):
    """A SimulationResult with no final_decision reports solution_found=False."""
    vf = ValueFunctionFactory.create_random_linear(["profit", "cost"], maximize=True, seed=42)
    result = SimulationResult(num_eval_sols=0)

    metrics = PerformanceEvaluator(_make_instance(small_pareto_front)).evaluate(result, vf)

    assert not metrics.solution_found
    assert metrics.found_value is None
    assert metrics.found_value_rank is None
    assert metrics.value_gap is None
    assert metrics.is_optimal is None


def test_nonlinear_value_function_optimal_ranking(small_pareto_front):
    """A NonLinear (Cobb-Douglas) value function also yields rank 1 for the best Pareto point."""
    vf = ValueFunctionFactory.create_random_nonlinear(["profit", "cost"], maximize=True, seed=7)
    best_point = max(small_pareto_front, key=vf.evaluate)

    metrics = PerformanceEvaluator(_make_instance(small_pareto_front)).evaluate(_wrap_solution(best_point), vf)

    assert metrics.solution_found
    assert metrics.is_optimal
    assert metrics.found_value_rank == 1
