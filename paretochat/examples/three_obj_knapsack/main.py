"""Implementation of a simple knapsack problem solver using OR-Tools."""

import os
from pathlib import Path
from typing import Annotated

from ortools.linear_solver import pywraplp
from pydantic import BaseModel
import typer
import yaml

from paretochat.api.cli.interface import get_approval_handler, interactive_main
from paretochat.domain.agents.agent import ParetoChatAgent
from paretochat.domain.algorithm import Algorithm
from paretochat.domain.problem import AlgorithmInfo, ObjectiveSense, ProblemInfo, ScalarizationKind, Solution
from paretochat.domain.scoring import ObjectiveScalarization, ScoringModel


class KnapsackInstance(BaseModel):
    """A class representing a knapsack problem instance."""

    num_objectives: int
    num_items: int
    capacity: int
    items: list[tuple[int, list[int]]]  # List of (weight, [obj1, obj2, obj3]) for each item


class KnapsackSolver(Algorithm):
    """A class representing the knapsack problem solver."""

    _BINARY_SOLUTION_THRESHOLD = 0.5

    def __init__(self):
        """Initialize the knapsack problem solver."""
        super().__init__()
        self.base_path = Path(__file__).parent
        self.info_path = self.base_path / "info.yaml"
        self.instance_path = self.base_path / "instance.txt"
        self._problem_info = self._load_problem_info()
        self._instance = self._load_instance()
        self._solver = self._setup_solver()
        self._x = {}  # Item decision variables
        self._z = {}  # Objective value variables
        self._build_model()
        self._scoring_model = ScoringModel(self._problem_info, self.algorithm_info, self._solver, self._z)

    def _load_problem_info(self) -> ProblemInfo:
        """Load problem information from info.yaml."""
        with open(self.info_path, "r") as f:
            info = yaml.safe_load(f)

        return ProblemInfo.model_validate(info.get("problem", {}))

    def _setup_solver(self) -> pywraplp.Solver:
        """Create and return the OR-Tools solver instance."""
        solver = pywraplp.Solver.CreateSolver("HIGHS")
        if not solver:
            raise RuntimeError("HIGHS solver not available in the installed ortools version.")
        return solver

    def _build_model(self):
        """Build the optimization model."""
        solver = self._solver
        items = self._instance.items
        capacity = self._instance.capacity
        # Variables: x[i] is 1 if item i is selected, 0 otherwise
        x = {}
        for i in range(len(items)):
            x[i] = solver.BoolVar(f"x_{i}")
        self._x = x  # Store decision variables for later use

        z = {}
        for idx, obj in enumerate(self._problem_info.objectives):
            z[obj.name] = solver.NumVar(float("-inf"), float("inf"), f"z_{idx}")
        self._z = z  # Store objective value variables for later use

        # Capacity constraint:
        # sum(weight_i * x_i) <= capacity
        solver.Add(solver.Sum(items[i][0] * x[i] for i in range(len(items))) <= capacity)

        # Constraints to link z variables to the actual objective values based on selected items
        for idx, obj in enumerate(self._problem_info.objectives):
            # z[idx] = sum(obj_value_i * x_i) for all items
            solver.Add(z[obj.name] == solver.Sum(items[i][1][idx] * x[i] for i in range(len(items))))

    def _print_model(self):
        """Print the model in LP format."""
        print(self._solver.ExportModelAsLpFormat(False))

    def _load_instance(self) -> KnapsackInstance:
        """Load the knapsack problem instance from instance.txt."""
        with open(self.instance_path, "r") as f:
            lines = f.readlines()

        num_items = int(lines[0].strip().split()[0])
        num_objectives = int(lines[0].strip().split()[1])
        capacity = int(lines[1].strip())

        items = []
        for i in range(num_items):
            parts = list(map(int, lines[2 + i].strip().split()))
            weight = parts[0]
            obj_values = parts[1 : 1 + num_objectives]
            items.append((weight, obj_values))

        return KnapsackInstance(num_objectives=num_objectives, num_items=num_items, capacity=capacity, items=items)

    def _solve(self):
        """Solve the optimization problem and return the solution."""
        # self._print_model()  # Print the model for debugging
        # self._solver.EnableOutput()  # Enable solver output for debugging
        status = self._solver.Solve()

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            # Extract Solution
            chosen_indices = [
                i
                for i in range(len(self._instance.items))
                if self._x[i].solution_value() > KnapsackSolver._BINARY_SOLUTION_THRESHOLD
            ]

            # Calculate final objective values (raw sums per objective)
            final_obj_values = {}
            for idx, obj_def in enumerate(self._problem_info.objectives):
                name = obj_def.name
                # Sum the values of chosen items for this specific objective column
                total = sum(self._instance.items[i][1][idx] for i in chosen_indices)
                final_obj_values[name] = float(total)

            return Solution(objective_values=final_obj_values)
        else:
            print(f"No feasible solution found: Solver status: {status}")
            return None

    def _lexicographic_solve(self, objective_scalarization: ObjectiveScalarization):
        """Solve the optimization problem using lexicographic scalarization."""
        obj_to_priority = objective_scalarization.get_params_dict().get("priorities", {})
        if not obj_to_priority:
            raise ValueError(
                "Lexicographic scalarization requires 'priorities' parameter mapping objective names to their priority levels."
            )

        priorities = set(obj_to_priority.values())
        unspecified_objs = [obj.name for obj in self._problem_info.objectives if obj.name not in obj_to_priority]
        if unspecified_objs:
            priorities.add(
                max(priorities) + 1 if priorities else 0
            )  # Assign a new priority level for unspecified objectives
        for obj_name in unspecified_objs:
            print(
                f"Warning: Objective '{obj_name}' does not have a specified priority in the lexicographic scalarization. It will be assigned the lowest priority."
            )
            obj_to_priority[obj_name] = max(priorities)

        previous_solution = None
        previous_objectives = None
        for priority in priorities:
            # Update the model for the current priority objectives
            current_objectives = [
                obj_name for obj_name, obj_priority in obj_to_priority.items() if obj_priority == priority
            ]
            self._scoring_model.update_model(
                objective_scalarization,
                current_level_objectives=current_objectives,
                previous_level_objectives=previous_objectives,
                previous_solution=previous_solution,
            )
            solution = self._solve()
            if solution is None:
                return None  # No feasible solution found at this level
            previous_solution = solution  # Update previous solution for the next iteration
            previous_objectives = current_objectives  # Update previous objectives for the next iteration

        return previous_solution  # Return the final solution after all iterations

    @property
    def algorithm_info(self) -> AlgorithmInfo:
        """Return the algorithm info."""
        return AlgorithmInfo(
            name="KnapsackSolver",
            description="A single objective knapsack problem solver using OR-Tools.",
            optimization_direction=ObjectiveSense.MAXIMIZE,
            supported_scalarizations=[
                ScalarizationKind.WEIGHTED_SUM,
                ScalarizationKind.REFERENCE_POINT,
                ScalarizationKind.LEXICOGRAPHIC,
            ],
            supports_obj_bounds=True,
        )

    def run(self, objective_scalarization: ObjectiveScalarization) -> Solution | None:
        """Run the algorithm with the given objective scalarization."""

        if objective_scalarization.is_lexicographic():
            solution = self._lexicographic_solve(objective_scalarization)
        else:
            self._scoring_model.update_model(objective_scalarization)
            solution = self._solve()
        return solution


def main(
    auto_approve: Annotated[
        bool,
        typer.Option(help="Run the algorithm without asking for approval. Use --no-auto-approve to confirm each run."),
    ] = True,
):
    domain_info_path = os.path.join(os.path.dirname(__file__), "info.yaml")
    algorithm = KnapsackSolver()
    agent = ParetoChatAgent(domain_info_path, algorithm, approval_handler=get_approval_handler(auto_approve))
    example_queries = [
        "Find a solution that maximizes the total objective value.",
        "Find a solution that balances all objectives equally.",
        "Find a solution that prioritizes the total value of items selected.",
    ]
    interactive_main(agent, example_queries)


if __name__ == "__main__":
    typer.run(main)
