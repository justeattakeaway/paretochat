import pytest

from paretochat.domain.problem import (
    AlgorithmInfo,
    ObjectiveInfo,
    ObjectiveSense,
    ProblemInfo,
    ScalarizationKind,
)
from paretochat.domain.scoring import (
    ObjectiveParameter,
    ObjectiveParameterKind,
    ObjectiveScalarization,
)


@pytest.fixture
def two_obj_problem_info():
    return ProblemInfo(
        description="Two-objective test problem",
        objectives=[
            ObjectiveInfo(name="profit", description="Profit to maximize", sense=ObjectiveSense.MAXIMIZE),
            ObjectiveInfo(name="cost", description="Cost to minimize", sense=ObjectiveSense.MINIMIZE),
        ],
    )


@pytest.fixture
def maximize_algorithm_info():
    return AlgorithmInfo(
        name="test-maximizer",
        description="Test algorithm",
        optimization_direction=ObjectiveSense.MAXIMIZE,
        supported_scalarizations=[
            ScalarizationKind.WEIGHTED_SUM,
            ScalarizationKind.REFERENCE_POINT,
            ScalarizationKind.LEXICOGRAPHIC,
        ],
        supports_obj_bounds=True,
    )


@pytest.fixture
def minimize_algorithm_info():
    return AlgorithmInfo(
        name="test-minimizer",
        description="Test algorithm",
        optimization_direction=ObjectiveSense.MINIMIZE,
        supported_scalarizations=[
            ScalarizationKind.WEIGHTED_SUM,
            ScalarizationKind.REFERENCE_POINT,
            ScalarizationKind.LEXICOGRAPHIC,
        ],
        supports_obj_bounds=True,
    )


@pytest.fixture
def small_pareto_front():
    return [
        {"profit": 100.0, "cost": 10.0},
        {"profit": 80.0, "cost": 8.0},
        {"profit": 60.0, "cost": 6.0},
        {"profit": 40.0, "cost": 4.0},
        {"profit": 20.0, "cost": 2.0},
    ]


@pytest.fixture
def default_scalarization():
    return ObjectiveScalarization(
        kind=ScalarizationKind.WEIGHTED_SUM,
        parameters=[
            ObjectiveParameter(objective_name="profit", kind=ObjectiveParameterKind.WEIGHT, value=0.5),
            ObjectiveParameter(objective_name="cost", kind=ObjectiveParameterKind.WEIGHT, value=0.5),
        ],
    )
