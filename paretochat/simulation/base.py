"""Base module for simulation data models."""

from enum import StrEnum


class ProblemType(StrEnum):
    KNAPSACK = "knapsack"
    SPA = "spa"


class SolutionTool(StrEnum):
    PARETOCHAT = "paretochat"
    DESDEO_RPM = "desdeo_rpm"
    DESDEO_NIMBUS = "desdeo_nimbus"


class ValueFunctionType(StrEnum):
    ALL = "all"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"
    SUM_SQUARED_DISTANCE = "sum_squared_distance"
