"""Tests for ScoringFunction workflows."""

import pytest

from paretochat.domain.problem import ObjectiveInfo, ObjectiveSense, ProblemInfo, ScalarizationKind
from paretochat.domain.scoring import (
    ObjectiveParameter,
    ObjectiveParameterKind,
    ObjectiveScalarization,
    ScoringFunction,
)


def test_weighted_sum_picks_best_solution(two_obj_problem_info, maximize_algorithm_info):
    """ScoringFunction correctly ranks three candidates under a weighted-sum scalarization."""
    # profit sense=MAXIMIZE matches optimization_direction=MAXIMIZE → weight stays positive
    # cost sense=MINIMIZE ≠ MAXIMIZE → weight is negated
    # score = 0.7*profit + (-0.3)*cost
    scalarization = ObjectiveScalarization(
        kind=ScalarizationKind.WEIGHTED_SUM,
        parameters=[
            ObjectiveParameter(objective_name="profit", kind=ObjectiveParameterKind.WEIGHT, value=0.7),
            ObjectiveParameter(objective_name="cost", kind=ObjectiveParameterKind.WEIGHT, value=0.3),
        ],
    )
    sf = ScoringFunction(two_obj_problem_info, maximize_algorithm_info, scalarization)

    score_a = sf.score(objective_values={"profit": 100.0, "cost": 50.0})  # 0.7*100 - 0.3*50 = 55
    score_b = sf.score(objective_values={"profit": 70.0, "cost": 30.0})  # 0.7*70 - 0.3*30 = 40
    score_c = sf.score(objective_values={"profit": 30.0, "cost": 10.0})  # 0.7*30 - 0.3*10 = 18

    assert score_a > score_b > score_c


def test_weighted_sum_respects_sense(two_obj_problem_info, maximize_algorithm_info):
    """Lower cost improves score when profit is held constant."""
    scalarization = ObjectiveScalarization(
        kind=ScalarizationKind.WEIGHTED_SUM,
        parameters=[
            ObjectiveParameter(objective_name="profit", kind=ObjectiveParameterKind.WEIGHT, value=0.5),
            ObjectiveParameter(objective_name="cost", kind=ObjectiveParameterKind.WEIGHT, value=0.5),
        ],
    )
    sf = ScoringFunction(two_obj_problem_info, maximize_algorithm_info, scalarization)

    score_low_cost = sf.score(objective_values={"profit": 50.0, "cost": 10.0})
    score_high_cost = sf.score(objective_values={"profit": 50.0, "cost": 30.0})

    assert score_low_cost > score_high_cost


def test_reference_point_favours_closer_solution(two_obj_problem_info, maximize_algorithm_info):
    """A solution close to the reference point scores better than one far away."""
    scalarization = ObjectiveScalarization(
        kind=ScalarizationKind.REFERENCE_POINT,
        parameters=[
            ObjectiveParameter(objective_name="profit", kind=ObjectiveParameterKind.TARGET, value=80.0),
            ObjectiveParameter(objective_name="cost", kind=ObjectiveParameterKind.TARGET, value=20.0),
            ObjectiveParameter(objective_name="profit", kind=ObjectiveParameterKind.WEIGHT, value=0.5),
            ObjectiveParameter(objective_name="cost", kind=ObjectiveParameterKind.WEIGHT, value=0.5),
        ],
    )
    sf = ScoringFunction(two_obj_problem_info, maximize_algorithm_info, scalarization)

    score_close = sf.score(objective_values={"profit": 79.0, "cost": 21.0})
    score_far = sf.score(objective_values={"profit": 20.0, "cost": 80.0})

    assert score_close > score_far


def test_lexicographic_score_returns_flat_list(two_obj_problem_info, maximize_algorithm_info):
    """Lexicographic scoring returns a flat list with one scalar per priority level."""
    scalarization = ObjectiveScalarization(
        kind=ScalarizationKind.LEXICOGRAPHIC,
        parameters=[
            ObjectiveParameter(objective_name="profit", kind=ObjectiveParameterKind.PRIORITY, value=1),
            ObjectiveParameter(objective_name="cost", kind=ObjectiveParameterKind.PRIORITY, value=2),
        ],
    )
    sf = ScoringFunction(two_obj_problem_info, maximize_algorithm_info, scalarization)

    result = sf.score(objective_values={"profit": 80.0, "cost": 30.0})

    assert isinstance(result, list)
    assert len(result) == 2  # two priority levels
    assert all(isinstance(v, (int, float)) for v in result)
    # priority 1 (profit, MAXIMIZE = same sense → weight +1.0): 1.0 * 80.0 = 80.0
    # priority 2 (cost, MINIMIZE ≠ MAXIMIZE → weight -1.0): -1.0 * 30.0 = -30.0
    assert result[0] == pytest.approx(80.0)
    assert result[1] == pytest.approx(-30.0)


def test_lexicographic_shared_priority_level_sums_objectives(maximize_algorithm_info):
    """Objectives at the same priority level are summed into one score for that level."""
    # Three objectives: profit and revenue share priority 1, cost gets priority 2.
    problem_info = ProblemInfo(
        description="Three-objective test problem",
        objectives=[
            ObjectiveInfo(name="profit", description="Profit", sense=ObjectiveSense.MAXIMIZE),
            ObjectiveInfo(name="revenue", description="Revenue", sense=ObjectiveSense.MAXIMIZE),
            ObjectiveInfo(name="cost", description="Cost", sense=ObjectiveSense.MINIMIZE),
        ],
    )
    scalarization = ObjectiveScalarization(
        kind=ScalarizationKind.LEXICOGRAPHIC,
        parameters=[
            ObjectiveParameter(objective_name="profit", kind=ObjectiveParameterKind.PRIORITY, value=1),
            ObjectiveParameter(objective_name="revenue", kind=ObjectiveParameterKind.PRIORITY, value=1),
            ObjectiveParameter(objective_name="cost", kind=ObjectiveParameterKind.PRIORITY, value=2),
        ],
    )
    sf = ScoringFunction(problem_info, maximize_algorithm_info, scalarization)

    result = sf.score(objective_values={"profit": 80.0, "revenue": 50.0, "cost": 30.0})

    # Two priority levels, not three
    assert isinstance(result, list)
    assert len(result) == 2
    # priority 1: profit (weight +1.0) + revenue (weight +1.0) = 80.0 + 50.0 = 130.0
    # priority 2: cost (weight -1.0) = -30.0
    assert result[0] == pytest.approx(130.0)
    assert result[1] == pytest.approx(-30.0)


def test_score_raises_without_scalarization(two_obj_problem_info, maximize_algorithm_info):
    """ScoringFunction.score() raises ValueError when no scalarization is configured."""
    sf = ScoringFunction(two_obj_problem_info, maximize_algorithm_info)

    with pytest.raises(ValueError):
        sf.score(objective_values={"profit": 50.0, "cost": 20.0})
