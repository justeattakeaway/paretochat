"""Data models and solvers needed for simulating the SPA (set partitioning) problem."""

from pathlib import Path
from typing import ClassVar

from ortools.linear_solver import pywraplp
from pydantic import BaseModel
import yaml

from paretochat.domain.algorithm import Algorithm
from paretochat.domain.problem import AlgorithmInfo, DomainInfo, ObjectiveSense, ScalarizationKind, Solution
from paretochat.domain.scoring import ObjectiveScalarization, ScoringModel
from paretochat.simulation.simulation_problem import SimulationProblem, SimulationProblemInstance


class SPASimulationProblem(SimulationProblem):
    """Simulation wrapper for the set partitioning (SPA) problem."""

    PARETO_FILE_PREFIX: ClassVar[str] = "Y_N_"

    def __init__(self, name: str, base_path: str | Path, max_instances: int = 10):
        base_path = Path(base_path)
        domain_info_path = base_path / "info.yaml"
        domain_info = self._load_domain_info(domain_info_path)
        instances = self._load_instances(base_path, domain_info, max_instances)
        super().__init__(
            name=name, domain_info_path=str(domain_info_path), domain_info=domain_info, instances=instances
        )

    def _load_domain_info(self, info_path: str | Path) -> DomainInfo:
        """Load domain info from the info.yaml file."""
        with open(info_path, "r") as f:
            config = yaml.safe_load(f)
        return DomainInfo.model_validate(config)

    def _to_pareto_file_name(self, instance_name: str) -> str:
        """Translate instance file names to the vOptLib nondominated file naming convention."""
        if instance_name.startswith("bio"):
            return f"{self.PARETO_FILE_PREFIX}{instance_name[3:]}.txt"
        return f"{self.PARETO_FILE_PREFIX}{instance_name}.txt"

    def _load_pareto_front(
        self,
        nondominated_dir: Path,
        instance_name: str,
        objective_names: list[str],
    ) -> list[dict[str, float]] | None:
        """Load the nondominated set for an instance; return None when not available."""
        pareto_file = nondominated_dir / self._to_pareto_file_name(instance_name)
        if not pareto_file.exists():
            return None

        with open(pareto_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) < 2:
            return None

        num_non_dominated = int(lines[1])
        pareto_front = []
        for point_line in lines[2 : 2 + num_non_dominated]:
            values = list(map(float, point_line.split()))
            pareto_front.append(dict(zip(objective_names, values, strict=True)))

        return pareto_front

    def _load_instances(
        self, base_path: str | Path, domain_info: DomainInfo, max_instances: int
    ) -> list[SimulationProblemInstance]:
        base_path = Path(base_path)
        instances_dir = base_path / "instances"
        nondominated_dir = base_path / "nondominated_sets"
        objective_names = [objective.name for objective in domain_info.problem.objectives]
        num_objectives = len(objective_names)

        instances: list[SimulationProblemInstance] = []
        for instance_path in sorted(instances_dir.glob("*.txt")):
            instance_name = instance_path.stem
            pareto_front = self._load_pareto_front(nondominated_dir, instance_name, objective_names)
            if pareto_front is None:
                # Skip instances that do not have a corresponding nondominated-set file.
                continue

            with open(instance_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]

            num_rows, num_columns = map(int, lines[0].split())
            expected_lines = 1 + num_columns
            if len(lines) < expected_lines:
                raise ValueError(
                    f"Instance '{instance_name}' is malformed: expected {expected_lines} lines, got {len(lines)}."
                )

            column_costs: list[list[float]] = []
            covered_rows_by_column: list[list[int]] = []
            for column_line in lines[1 : 1 + num_columns]:
                parts = column_line.split()
                min_length = num_objectives + 1
                if len(parts) < min_length:
                    raise ValueError(f"Malformed SPA column entry in '{instance_name}': '{column_line}'")

                costs = [float(parts[idx]) for idx in range(num_objectives)]
                num_covered_rows = int(parts[num_objectives])
                row_values = [int(v) for v in parts[num_objectives + 1 :]]
                if len(row_values) != num_covered_rows:
                    raise ValueError(
                        f"Malformed covered-row list in '{instance_name}': "
                        f"expected {num_covered_rows}, got {len(row_values)}."
                    )

                # Convert 1-based row ids from files into 0-based row indices used internally.
                zero_based_rows = [row_id - 1 for row_id in row_values]
                if any(row_idx < 0 or row_idx >= num_rows for row_idx in zero_based_rows):
                    raise ValueError(
                        f"Out-of-range row index found in '{instance_name}' with m={num_rows}: {row_values}"
                    )

                column_costs.append(costs)
                covered_rows_by_column.append(zero_based_rows)

            instances.append(
                SimulationProblemInstance(
                    name=instance_name,
                    input_data=SPAInstance(
                        num_rows=num_rows,
                        num_columns=num_columns,
                        num_objectives=num_objectives,
                        column_costs=column_costs,
                        covered_rows_by_column=covered_rows_by_column,
                    ),
                    pareto_front=pareto_front,
                )
            )

            if len(instances) >= max_instances:
                break

        return instances


class SPAInstance(BaseModel):
    """A class representing one SPA (set partitioning) instance."""

    num_rows: int
    num_columns: int
    num_objectives: int
    column_costs: list[list[float]]
    covered_rows_by_column: list[list[int]]


class SPASolver(Algorithm):
    """A class representing the SPA solver."""

    _BINARY_SOLUTION_THRESHOLD = 0.5

    def __init__(self, domain_info: DomainInfo, instance: SPAInstance):
        """Initialize the SPA solver."""
        super().__init__()
        self._instance = instance
        self._problem_info = domain_info.problem
        self._solver = self._setup_solver()
        self._x = {}  # Column decision variables
        self._z = {}  # Objective value variables
        self._build_model()
        self._scoring_model = ScoringModel(self._problem_info, self.algorithm_info, self._solver, self._z)

    def _setup_solver(self) -> pywraplp.Solver:
        """Create and return the OR-Tools solver instance."""
        solver = pywraplp.Solver.CreateSolver("GUROBI")
        if not solver:
            raise RuntimeError("Gurobi solver not available in the installed ortools version.")
        return solver

    def _build_model(self):
        """Build the SPA optimization model."""
        solver = self._solver
        num_columns = self._instance.num_columns
        num_rows = self._instance.num_rows

        x = {}
        for column_idx in range(num_columns):
            x[column_idx] = solver.BoolVar(f"x_{column_idx}")
        self._x = x

        z = {}
        for obj_idx, obj in enumerate(self._problem_info.objectives):
            z[obj.name] = solver.NumVar(float("-inf"), float("inf"), f"z_{obj_idx}")
        self._z = z

        # Set-partitioning constraints: each row must be covered by exactly one selected column.
        for row_idx in range(num_rows):
            covering_columns = [
                column_idx
                for column_idx, covered_rows in enumerate(self._instance.covered_rows_by_column)
                if row_idx in covered_rows
            ]
            if not covering_columns:
                raise ValueError(f"Row {row_idx} has no covering column; instance is infeasible by construction.")
            solver.Add(solver.Sum(x[column_idx] for column_idx in covering_columns) == 1)

        for obj_idx, obj in enumerate(self._problem_info.objectives):
            solver.Add(
                z[obj.name]
                == solver.Sum(
                    self._instance.column_costs[column_idx][obj_idx] * x[column_idx]
                    for column_idx in range(num_columns)
                )
            )

    def _solve(self):
        """Solve the optimization problem and return the solution."""
        status = self._solver.Solve()

        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            chosen_columns = [
                column_idx
                for column_idx in range(self._instance.num_columns)
                if self._x[column_idx].solution_value() > SPASolver._BINARY_SOLUTION_THRESHOLD
            ]

            final_obj_values = {}
            for obj_idx, obj_def in enumerate(self._problem_info.objectives):
                total = sum(self._instance.column_costs[column_idx][obj_idx] for column_idx in chosen_columns)
                final_obj_values[obj_def.name] = float(total)

            return Solution(objective_values=final_obj_values)

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
            priorities.add(max(priorities) + 1 if priorities else 0)
        for obj_name in unspecified_objs:
            print(
                f"Warning: Objective '{obj_name}' does not have a specified priority in the lexicographic scalarization. It will be assigned the lowest priority."
            )
            obj_to_priority[obj_name] = max(priorities)

        previous_solution = None
        previous_objectives = None
        for priority in sorted(priorities):
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
                return None
            previous_solution = solution
            previous_objectives = current_objectives

        return previous_solution

    @property
    def algorithm_info(self) -> AlgorithmInfo:
        """Return the algorithm info."""
        return AlgorithmInfo(
            name="SPASolver",
            description="A set partitioning solver using OR-Tools.",
            optimization_direction=ObjectiveSense.MINIMIZE,
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
            return self._lexicographic_solve(objective_scalarization)

        self._scoring_model.update_model(objective_scalarization)
        return self._solve()
