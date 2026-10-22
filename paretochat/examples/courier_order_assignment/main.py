"""Demonstration of ParetoChat on the Courier Order Assignment problem."""

import csv
import json
import os
from pathlib import Path
import subprocess  # nosec B404
from typing import Annotated, Optional

import typer

from paretochat.api.cli.interface import get_approval_handler, interactive_main
from paretochat.domain.agents.agent import ParetoChatAgent
from paretochat.domain.algorithm import Algorithm
from paretochat.domain.problem import AlgorithmInfo, ObjectiveSense, Solution
from paretochat.domain.scoring import (
    ObjectiveParameter,
    ObjectiveParameterKind,
    ObjectiveScalarization,
    ScalarizationKind,
)

_DEFAULT_JAR = "solver-analytics-1.0.0-SNAPSHOT-jar-with-dependencies.jar"


class CourierOrderAssignment(Algorithm):
    def __init__(self, instance_dir_path: str, output_csv_path: str, jar_path: str | None = None) -> None:
        super().__init__()
        base_dir = Path(__file__).resolve().parent
        self.instance_dir_path = str(self._resolve_path(base_dir, instance_dir_path))
        self.output_csv_path = str(self._resolve_path(base_dir, output_csv_path))
        self.jar_path = str(self._resolve_path(base_dir, jar_path if jar_path is not None else _DEFAULT_JAR))

    @staticmethod
    def _resolve_path(base_dir: Path, path: str) -> Path:
        resolved_path = Path(path)
        if resolved_path.is_absolute():
            return resolved_path
        return base_dir / resolved_path

    @property
    def algorithm_info(self):
        return AlgorithmInfo(
            name="Courier Order Assignment",
            description="Assigns orders to couriers while optimizing multiple criteria.",
            optimization_direction=ObjectiveSense.MINIMIZE,
            supported_scalarizations=[
                ScalarizationKind.WEIGHTED_SUM,
                ScalarizationKind.LEXICOGRAPHIC,
            ],
            supports_obj_bounds=False,
        )

    def run(self, objective_scalarization: ObjectiveScalarization) -> Solution | None:
        self._update_criteria(objective_scalarization)
        self._invoke_solver()
        solutions = self._read_solutions()
        if not solutions:
            return None

        objective_sums: dict[str, float] = {}
        objective_counts: dict[str, int] = {}

        for solution in solutions.values():
            for objective_name, objective_value in solution.objective_values.items():
                objective_sums[objective_name] = objective_sums.get(objective_name, 0.0) + objective_value
                objective_counts[objective_name] = objective_counts.get(objective_name, 0) + 1

        if not objective_sums:
            return None

        averaged_objective_values = {
            objective_name: objective_sums[objective_name] / objective_counts[objective_name]
            for objective_name in objective_sums
        }
        return Solution(objective_values=averaged_objective_values)

    def _update_criteria(self, objective_scalarization: ObjectiveScalarization) -> None:
        params = objective_scalarization.get_params_dict()
        weights = params.get("weights", {})
        priorities = params.get("priorities", {})

        # bounds are ignored

        instance_dir = Path(self.instance_dir_path)
        for instance_file_path in instance_dir.glob("*.json"):
            with open(instance_file_path, encoding="utf-8") as instance_file:
                try:
                    instance_data = json.load(instance_file)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Instance file {instance_file_path} contains invalid JSON.") from exc

            instance_obj = instance_data[-1] if isinstance(instance_data, list) and instance_data else instance_data
            minimization_objective = (
                instance_obj.get("minimizationObjective") if isinstance(instance_obj, dict) else None
            )
            criteria = minimization_objective.get("criteria") if isinstance(minimization_objective, dict) else None
            if not isinstance(criteria, list):
                raise ValueError("Instance file is missing minimizationObjective.criteria.")

            for criterion in criteria:
                if not isinstance(criterion, dict):
                    continue

                criterion_name = criterion.get("name")
                if criterion_name in priorities:
                    # For AssignmentProblemSolver, a higher number indicates higher priority, being opposite to how Agent interprets it.
                    criterion["priority"] = max(priorities.values()) - priorities[criterion_name] + 1
                else:
                    criterion["priority"] = 1

                # have a default weight of 1 for criteria not specified in the scalarization parameters to avoid dominated points
                criterion["weight"] = weights.get(criterion_name, 1)

            with open(instance_file_path, "w", encoding="utf-8") as instance_file:
                json.dump(instance_data, instance_file, indent=2)
                instance_file.write("\n")

    def _invoke_solver(self) -> None:
        subprocess.run(  # nosec B603, B607
            [
                "java",
                "-jar",
                self.jar_path,
                self.instance_dir_path,
                self.output_csv_path,
                "false",
            ],
            check=True,
        )

    def _read_solutions(self) -> dict[str, Solution]:
        instance_dir = Path(self.instance_dir_path)
        key_to_path: dict[tuple[str, str], str] = {}
        for instance_file in instance_dir.glob("*.json"):
            with open(instance_file, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Instance file {instance_file} contains invalid JSON.") from exc
            region = data[0].get("regionPrefix", "") if isinstance(data, list) and data else ""
            instance_obj = data[-1] if isinstance(data, list) else data
            timestamp = instance_obj.get("timestamp", "") if isinstance(instance_obj, dict) else ""
            if region and timestamp:
                key_to_path[(region, timestamp)] = str(instance_file)

        results: dict[str, Solution] = {}
        with open(self.output_csv_path, encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                region = row.get("region", "").strip()
                timestamp = row.get("timestamp", "").strip()
                criterion_value_str = row.get("solution.cost.criterionValue", "").strip()

                instance_file_path = key_to_path.get((region, timestamp))
                if instance_file_path is None:
                    continue

                objective_values: dict[str, float] = {}
                for entry_ in criterion_value_str.split("|"):
                    entry = entry_.strip()
                    if "=[" not in entry:
                        continue
                    name, rest = entry.split("=[", 1)
                    first_value_str = rest.rstrip("]").split(",")[0].strip()
                    try:
                        objective_values[name] = float(first_value_str)
                    except ValueError:
                        pass

                results[instance_file_path] = Solution(objective_values=objective_values)

        return results


def algorithm_run_example():
    algorithm = CourierOrderAssignment(
        instance_dir_path="instance",
        output_csv_path="run.csv",
    )
    scalarization = ObjectiveScalarization(
        kind=ScalarizationKind.WEIGHTED_SUM,
        parameters=[
            ObjectiveParameter(
                objective_name="TotalPickupDurationCriterion",
                kind=ObjectiveParameterKind.WEIGHT,
                value=0.0,
            ),
            ObjectiveParameter(
                objective_name="TotalDeliveryDelayCriterion",
                kind=ObjectiveParameterKind.WEIGHT,
                value=1.0,
            ),
            ObjectiveParameter(
                objective_name="TotalPickupDelayCriterion", kind=ObjectiveParameterKind.WEIGHT, value=0.0
            ),
        ],
        bounds=[],
    )
    solution = algorithm.run(scalarization)
    print("Objective Values:", solution.objective_values if solution else "No solution found.")


def main(
    instance_dir: Annotated[
        str, typer.Option(help="Path to the directory containing courier instance JSON files.")
    ] = "instance",
    output_csv: Annotated[str, typer.Option(help="Path for the solver output CSV file.")] = "run.csv",
    jar_path: Annotated[
        Optional[str], typer.Option(help="Path to the solver JAR. Defaults to the JAR bundled alongside this script.")
    ] = None,
    auto_approve: Annotated[
        bool,
        typer.Option(help="Run the algorithm without asking for approval. Use --no-auto-approve to confirm each run."),
    ] = True,
):
    domain_info_path = os.path.join(os.path.dirname(__file__), "info.yaml")
    algorithm = CourierOrderAssignment(
        instance_dir_path=instance_dir,
        output_csv_path=output_csv,
        jar_path=jar_path,
    )
    agent = ParetoChatAgent(domain_info_path, algorithm, approval_handler=get_approval_handler(auto_approve))
    example_queries = [
        "Find a solution that minimizes the total objective value.",
        "Find a solution that balances all objectives equally.",
        "Find a solution that prioritizes the total pickup duration.",
    ]
    interactive_main(agent, example_queries)


if __name__ == "__main__":
    typer.run(main)
