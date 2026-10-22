from typing import Optional

from pydantic import BaseModel

from paretochat.simulation.ai_dm import SimulationResult
from paretochat.simulation.simulation_problem import SimulationProblemInstance
from paretochat.simulation.value_function import ValueFunction


class EvaluationMetrics(BaseModel):
    pareto_size: int
    solution_found: bool
    found_value: Optional[float]
    found_value_rank: Optional[int]
    found_value_rank_normalized: Optional[float]
    best_possible_value: Optional[float]
    value_gap: Optional[float]
    value_ratio: Optional[float]
    is_optimal: Optional[bool]
    num_eval_sols: int


class PerformanceEvaluator:
    """Evaluates the performance of the AI Decision Maker."""

    def __init__(self, problem_instance: SimulationProblemInstance):
        self.problem_instance = problem_instance

    def evaluate(
        self, result: SimulationResult, value_function: ValueFunction, maximize: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate the result against the best possible solution in the Pareto front
        for the given value function.
        """

        solution_found = False
        found_value = None
        if result.final_decision:
            found_value = value_function.evaluate(result.final_decision.solution.objective_values)
            solution_found = True

        pareto_values = []
        best_possible_value = None
        nadir_point_value = None
        if self.problem_instance.pareto_front:
            for point in self.problem_instance.pareto_front:
                # value is always to be maximized, so higher is better.
                val = value_function.evaluate(point)
                pareto_values.append(val)
                if best_possible_value is None or val > best_possible_value:
                    best_possible_value = val
                if nadir_point_value is None or val < nadir_point_value:
                    nadir_point_value = val

        found_value_rank = None
        found_value_rank_normalized = None
        if pareto_values and found_value is not None:
            tolerance = 1e-9
            strictly_better_count = sum(val > found_value + tolerance for val in pareto_values)
            # Rank 1 means no Pareto-front point is strictly better than the found value.
            found_value_rank = strictly_better_count + 1
            found_value_rank = min(found_value_rank, len(pareto_values))  # Cap rank at pareto size

            pareto_size = len(pareto_values)
            # Normalize to [0, 1] where 1.0 is best rank and 0.0 is worst rank.
            if pareto_size == 1:
                found_value_rank_normalized = 1.0
            else:
                found_value_rank_normalized = 1.0 - ((found_value_rank - 1) / (pareto_size - 1))

        if best_possible_value is not None and found_value is not None:
            value_abs_gap = best_possible_value - found_value
            value_rel_gap = (
                value_abs_gap / abs(best_possible_value - nadir_point_value)
                if best_possible_value != nadir_point_value
                else 0.0
            )
            value_ratio = found_value / best_possible_value if best_possible_value != 0 else 0.0
            # Handle small floating point errors
            is_optimal = value_abs_gap < 1e-5
        else:
            value_abs_gap = None
            value_rel_gap = None
            value_ratio = None
            is_optimal = None

        return EvaluationMetrics(
            value_gap=value_rel_gap,
            value_ratio=value_ratio,
            num_eval_sols=result.num_eval_sols,
            found_value=found_value,
            found_value_rank=found_value_rank,
            found_value_rank_normalized=found_value_rank_normalized,
            best_possible_value=best_possible_value,
            is_optimal=is_optimal,
            solution_found=solution_found,
            pareto_size=len(self.problem_instance.pareto_front),
        )
