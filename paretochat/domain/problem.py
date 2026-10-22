"""Domain models for a problem."""

import ast
from enum import StrEnum
import math

from pydantic import BaseModel, Field, field_validator


class ObjectiveSense(StrEnum):
    """The sense of an objective."""

    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"


class ScalarizationKind(StrEnum):
    """The kind of scalarization."""

    LEXICOGRAPHIC = "LEXICOGRAPHIC"
    WEIGHTED_SUM = "WEIGHTED_SUM"
    REFERENCE_POINT = "REFERENCE_POINT"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_str(cls):
        return ", ".join(ScalarizationKind.list())


class ObjectiveInfo(BaseModel):
    """A class representing an objective of a problem."""

    name: str
    description: str
    sense: ObjectiveSense


class ProblemInfo(BaseModel):
    """A class representing a problem to be solved."""

    description: str
    objectives: list[ObjectiveInfo]


class AlgorithmInfo(BaseModel):
    """A class representing an algorithm to solve a problem."""

    name: str
    description: str
    optimization_direction: ObjectiveSense
    supported_scalarizations: list[ScalarizationKind] = Field(
        default_factory=lambda: [
            ScalarizationKind.WEIGHTED_SUM,
            ScalarizationKind.REFERENCE_POINT,
            ScalarizationKind.LEXICOGRAPHIC,
        ],
        description="The scalarization kinds the algorithm can handle.",
    )
    supports_obj_bounds: bool = Field(
        description="Whether the algorithm can handle objective bounds in the scalarization.",
    )


class DomainInfo(BaseModel):
    """A class representing the domain of a problem."""

    problem: ProblemInfo
    algorithm: AlgorithmInfo


class Solution(BaseModel):
    """A class representing a solution to a problem."""

    objective_values: dict[str, float] = Field(
        description="The objective values of the solution.",
    )

    @field_validator("objective_values", mode="before")
    @classmethod
    def parse_objective_values(cls, v):
        if isinstance(v, str):
            try:
                return ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass
        return v

    def obj_equals(self, other: "Solution") -> bool:
        """Check if the objective values of two solutions are equal."""
        if self.objective_values.keys() != other.objective_values.keys():
            return False

        for k, v in self.objective_values.items():
            if not math.isclose(v, other.objective_values[k], rel_tol=1e-4, abs_tol=1e-5):
                return False

        return True
