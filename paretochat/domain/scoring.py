"""Base class for scoring functions."""

from collections import defaultdict
from enum import Enum
import math
from typing import List, Optional

from ortools.linear_solver import pywraplp
from pydantic import BaseModel, Field

from paretochat.domain.problem import AlgorithmInfo, ObjectiveSense, ProblemInfo, ScalarizationKind, Solution
from paretochat.domain.scaling import ScalingFunction


class ObjectiveParameterKind(str, Enum):
    """The kind of objective parameter."""

    WEIGHT = "WEIGHT"
    PRIORITY = "PRIORITY"
    TARGET = "TARGET"


class ObjectiveParameter(BaseModel):
    """A parameter and its value associated with an objective."""

    objective_name: str = Field(description="The name of the objective.")
    kind: ObjectiveParameterKind = Field(description="The kind of the objective parameter.")
    value: float = Field(description="The numeric value of the objective parameter.")


class ObjectiveBound(BaseModel):
    """A bound for an objective."""

    objective_name: str = Field(description="The name of the objective.")
    min_value: Optional[float] = Field(default=None, description="The minimum bound value.")
    max_value: Optional[float] = Field(default=None, description="The maximum bound value.")


class ObjectiveScalarization(BaseModel):
    """A class representing the extracted scalarization from user new solution request."""

    kind: ScalarizationKind = Field(
        default=ScalarizationKind.WEIGHTED_SUM,
        description="The kind of the scalarization. E.g., LEXICOGRAPHIC, WEIGHTED_SUM, REFERENCE_POINT.",
    )

    parameters: list[ObjectiveParameter] = Field(
        default_factory=list,
        description="List of objective parameters and their values.",
    )

    bounds: list[ObjectiveBound] = Field(
        default_factory=list,
        description="List of bounds for each objective, if any.",
    )

    def get_params_dict(self) -> dict[str, dict[str, float]]:
        """Convert parameters to the dict format used internally."""
        return {
            "weights": {p.objective_name: p.value for p in self.parameters if p.kind == ObjectiveParameterKind.WEIGHT},
            "priorities": {
                p.objective_name: int(p.value) for p in self.parameters if p.kind == ObjectiveParameterKind.PRIORITY
            },
            "reference_point": {
                p.objective_name: p.value for p in self.parameters if p.kind == ObjectiveParameterKind.TARGET
            },
        }

    def get_bounds_dict(self) -> dict[str, tuple[float | None, float | None]]:
        """Convert bounds list to dict format."""
        return {b.objective_name: (b.min_value, b.max_value) for b in self.bounds}

    def is_lexicographic(self) -> bool:
        """Check if the scalarization is lexicographic."""
        return self.kind == ScalarizationKind.LEXICOGRAPHIC


class ScoringFunction:
    """Implement the scoring function to score a solution."""

    def __init__(
        self,
        problem_info: ProblemInfo,
        algorithm_info: AlgorithmInfo,
        scalarization: Optional[ObjectiveScalarization] = None,
        scaling_function: Optional[ScalingFunction] = None,
    ) -> None:
        """Initialize the scoring function."""
        self._problem_info = problem_info
        self._algorithm_info = algorithm_info
        self._scalarization = scalarization
        self._scaling_function = scaling_function

    def _scale_objective_values(self, objective_values: dict[str, float]) -> dict[str, float]:
        """Scale the objective values into the same range."""
        if self._scaling_function is None:
            return objective_values
        return self._scaling_function.transform(objective_values)

    def _get_weights(self) -> dict[str, float]:
        """Get the weights for each objective based on the scalarization parameters."""
        weights = self._scalarization.get_params_dict().get("weights", {})
        for obj in self._problem_info.objectives:
            w = weights.get(obj.name, 1.0)
            if obj.sense != self._algorithm_info.optimization_direction:
                w = -1 * w
            weights[obj.name] = w
        return weights

    def _apply_weighted_sum(self, objective_values: dict[str, float]) -> float:
        """Apply the weighted sum scalarization."""
        weights = self._get_weights()
        return sum(weights[o] * v for o, v in objective_values.items())

    def _apply_lexicographic(self, objective_values: dict[str, float]) -> float:
        """Apply the lexicographic scalarization."""
        priorities = self._scalarization.get_params_dict().get("priorities", {})
        weights = self._get_weights()
        # Assign a default lower priority (higher value) to objectives without explicit priority
        max_priority = max(priorities.values(), default=0)
        objectives_prioritized = sorted(
            objective_values.items(),
            key=lambda x: priorities.get(x[0], max_priority + 1),
        )
        lex_weights = [math.pow(10, -1 * i) for i in range(len(objectives_prioritized))]
        return sum(lex_weights[i] * weights[o] * v for i, (o, v) in enumerate(objectives_prioritized))

    def _apply_vectorized_lexicographic(self, objective_values: dict[str, float]) -> List[float]:
        """Apply the lexicographic scalarization and return a vector of scores for each objective."""
        priorities = self._scalarization.get_params_dict().get("priorities", {})
        weights = self._get_weights()
        max_priority = max(priorities.values(), default=1)
        unspecified_objectives = [o for o in objective_values if o not in priorities]
        for o in unspecified_objectives:
            priorities[o] = max_priority + 1
        priority_to_objectives = defaultdict(list)
        for o, p in priorities.items():
            priority_to_objectives[p].append(o)
        return [
            sum(weights[o] * objective_values[o] for o in priority_to_objectives[p])
            for p in sorted(priority_to_objectives)
        ]

    def _apply_reference_point(self, objective_values: dict[str, float]) -> float:
        """Apply the reference point scalarization."""
        reference_point = self._scalarization.get_params_dict().get("reference_point", {})
        reference_point = self._scale_objective_values(reference_point)
        weights = self._get_weights()
        # calculate the max difference between the objective values and the reference point, weighted by the weights
        obj_name_to_obj = {obj.name: obj for obj in self._problem_info.objectives}
        distance = 0.0
        if self._algorithm_info.optimization_direction == ObjectiveSense.MAXIMIZE:
            signed_obj_values = {
                o: v if obj_name_to_obj[o].sense == ObjectiveSense.MAXIMIZE else -v for o, v in objective_values.items()
            }
            distance = max(weights[o] * (reference_point.get(o, 0) - v) for o, v in signed_obj_values.items())
            distance = -distance
        else:
            signed_obj_values = {
                o: v if obj_name_to_obj[o].sense == ObjectiveSense.MINIMIZE else -v for o, v in objective_values.items()
            }
            distance = max(weights[o] * (v - reference_point.get(o, 0)) for o, v in signed_obj_values.items())
        # to prevent weekly nondominated solutions from having the same score, we can add a small-weighed sum of the objective values as a tie-breaker
        for o, v in objective_values.items():
            if obj_name_to_obj[o].sense == self._algorithm_info.optimization_direction:
                distance += 1e-4 * v
            else:
                distance -= 1e-4 * v

        return distance

    def score(
        self,
        scalarization: Optional[ObjectiveScalarization] = None,
        objective_values: dict[str, float] = None,
    ) -> float | List[float]:
        """Score the objective values."""
        # set the scalarization
        if scalarization is not None:
            self._scalarization = scalarization

        if self._scalarization is None:
            raise ValueError("Scalarization is not set for scoring function.")

        # scale the objective values
        objective_values = self._scale_objective_values(objective_values)

        if self._scalarization.kind == ScalarizationKind.LEXICOGRAPHIC:
            if ScalarizationKind.LEXICOGRAPHIC in self._algorithm_info.supported_scalarizations:
                return self._apply_vectorized_lexicographic(objective_values)
            else:
                return self._apply_lexicographic(objective_values)
        elif self._scalarization.kind == ScalarizationKind.WEIGHTED_SUM:
            return self._apply_weighted_sum(objective_values)
        elif self._scalarization.kind == ScalarizationKind.REFERENCE_POINT:
            return self._apply_reference_point(objective_values)
        else:
            raise ValueError(f"Invalid scalarization kind: {self._scalarization.kind}")


class ScoringModel:
    """A class responsible for updating the mathematical model with the scalarization constraints and objective."""

    def __init__(
        self,
        problem_info: ProblemInfo,
        algorithm_info: AlgorithmInfo,
        math_model: pywraplp.Solver,
        objective_vars: dict[str, pywraplp.Variable],
    ) -> None:
        self._problem_info = problem_info
        self._algorithm_info = algorithm_info
        self._math_model = math_model
        self._objective_vars = objective_vars
        self._ref_point_constraints = {}
        self._lex_opt_constraints = []

    def _reset_model(self, keep_lex_constraints=False):
        """Reset the model to its initial state (clearing out previously added additional vars and constraints)."""
        # clear the objective
        self._math_model.Objective().Clear()

        # clear reference point constraints
        for constr in self._ref_point_constraints.values():
            constr.Clear()
            constr.SetBounds(float("-inf"), float("inf"))
        self._ref_point_constraints = {}

        # clear lexicographic optimization constraints
        if not keep_lex_constraints:
            for constr in self._lex_opt_constraints:
                constr.Clear()
                constr.SetBounds(float("-inf"), float("inf"))
            self._lex_opt_constraints = []

        # clear objective variable bounds
        for obj_var in self._objective_vars.values():
            obj_var.SetBounds(float("-inf"), float("inf"))

    def _update_objective_bounds(self, objective_scalarization: ObjectiveScalarization):
        """Update the bounds of the objective variables based on the scalarization parameters."""
        bounds = objective_scalarization.get_bounds_dict()
        for obj_name, obj_var in self._objective_vars.items():
            current_lb = obj_var.lb()
            current_ub = obj_var.ub()
            new_lb, new_ub = bounds.get(obj_name, (None, None))
            if new_lb is not None and current_lb < new_lb:
                obj_var.SetLb(new_lb)
            if new_ub is not None and current_ub > new_ub:
                obj_var.SetUb(new_ub)

    def _get_weights(self, objective_scalarization: ObjectiveScalarization) -> dict[str, float]:
        """Get the weights for each objective based on the scalarization parameters."""
        weights = objective_scalarization.get_params_dict().get("weights", {})
        for obj in self._problem_info.objectives:
            w = weights.get(obj.name, 1.0)
            if obj.sense != self._algorithm_info.optimization_direction:
                w = -1 * w
            weights[obj.name] = w
        return weights

    def _update_for_weighted_sum(self, objective_scalarization: ObjectiveScalarization):
        """Update the model for weighted sum scalarization."""
        self._reset_model()
        weights = self._get_weights(objective_scalarization)

        obj_expr = self._math_model.Sum(
            weights[obj_name] * obj_var for obj_name, obj_var in self._objective_vars.items()
        )

        if self._algorithm_info.optimization_direction == ObjectiveSense.MAXIMIZE:
            self._math_model.Maximize(obj_expr)
        else:
            self._math_model.Minimize(obj_expr)

        self._update_objective_bounds(objective_scalarization)

    def _update_for_reference_point(self, objective_scalarization: ObjectiveScalarization):
        """Update the model for reference point scalarization."""
        self._reset_model()
        weights = self._get_weights(objective_scalarization)
        reference_point = objective_scalarization.get_params_dict().get("reference_point", {})
        d = self._math_model.NumVar(float("-inf"), float("inf"), "paretochat_var_distance")
        for obj in self._problem_info.objectives:
            w = weights[obj.name]
            r = reference_point.get(obj.name, 0.0)
            z = self._objective_vars[obj.name]
            constr_name = f"paretochat_constr_ref_point_{obj.name}"
            constr = self._math_model.Add(d >= w * (r - z), f"paretochat_constr_ref_point_{obj.name}")
            self._ref_point_constraints[constr_name] = constr
        self._math_model.Minimize(d)

        self._update_objective_bounds(objective_scalarization)

    def _update_for_lexicographic(
        self,
        objective_scalarization: ObjectiveScalarization,
        current_level_objectives: Optional[List[str]] = None,
        previous_level_objectives: Optional[List[str]] = None,
        previous_solution: Optional[Solution] = None,
    ):
        """Update the model for lexicographic scalarization."""
        if previous_level_objectives is not None:
            self._reset_model(keep_lex_constraints=True)
        else:
            self._reset_model()

        if current_level_objectives is None:
            raise ValueError("Current level objectives must be provided for lexicographic scalarization.")
        if previous_level_objectives is None and previous_solution is not None:
            raise ValueError("Previous level objectives must be provided for lexicographic scalarization.")
        if previous_solution is None and previous_level_objectives is not None:
            raise ValueError("Previous solution must be provided for lexicographic scalarization.")

        weights = self._get_weights(objective_scalarization)

        obj_expr = self._math_model.Sum(
            weights[obj_name] * self._objective_vars[obj_name] for obj_name in current_level_objectives
        )
        if self._algorithm_info.optimization_direction == ObjectiveSense.MAXIMIZE:
            self._math_model.Maximize(obj_expr)
        else:
            self._math_model.Minimize(obj_expr)

        if previous_level_objectives is not None and previous_solution is not None:
            constr_name = f"paretochat_constr_lex_{'_'.join(previous_level_objectives)}"
            constr_expr = self._math_model.Sum(
                weights[obj_name] * self._objective_vars[obj_name] for obj_name in previous_level_objectives
            )
            lex_constr = self._math_model.Add(float("-inf") <= constr_expr <= float("inf"), constr_name)
            obj_val = sum(
                weights[obj_name] * previous_solution.objective_values[obj_name]
                for obj_name in previous_level_objectives
            )
            if self._algorithm_info.optimization_direction == ObjectiveSense.MAXIMIZE:
                lex_constr = self._math_model.Add(constr_expr >= obj_val, constr_name)
            else:
                lex_constr = self._math_model.Add(constr_expr <= obj_val, constr_name)
            self._lex_opt_constraints.append(lex_constr)

        self._update_objective_bounds(objective_scalarization)

    def update_model(
        self,
        objective_scalarization: ObjectiveScalarization,
        current_level_objectives: Optional[List[str]] = None,
        previous_level_objectives: Optional[List[str]] = None,
        previous_solution: Optional[Solution] = None,
    ) -> pywraplp.Solver:
        """Updates the given mathematical model to incorporate the scalarization."""

        if objective_scalarization.kind == ScalarizationKind.WEIGHTED_SUM:
            self._update_for_weighted_sum(objective_scalarization)

        elif objective_scalarization.kind == ScalarizationKind.REFERENCE_POINT:
            self._update_for_reference_point(objective_scalarization)

        elif objective_scalarization.kind == ScalarizationKind.LEXICOGRAPHIC:
            self._update_for_lexicographic(
                objective_scalarization,
                current_level_objectives=current_level_objectives,
                previous_level_objectives=previous_level_objectives,
                previous_solution=previous_solution,
            )

        else:
            raise ValueError(f"Unsupported scalarization kind: {objective_scalarization.kind}")

        return self._math_model
