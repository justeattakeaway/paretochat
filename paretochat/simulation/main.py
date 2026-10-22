from datetime import datetime
from pathlib import Path

import typer

from paretochat.simulation.base import ProblemType, SolutionTool, ValueFunctionType
from paretochat.simulation.knapsack import KnapsackSimulationProblem
from paretochat.simulation.runner import run_simulation_batch
from paretochat.simulation.simulation_problem import SimulationProblem
from paretochat.simulation.spa import SPASimulationProblem

app = typer.Typer()


@app.callback()
def main() -> None:
    """Run simulation workflows from the command line."""


def _default_problem_name(problem_type: ProblemType, problem_path: Path) -> str:
    return f"{problem_type.value}-{problem_path.name}"


def _default_output_file() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"simulation_results_{timestamp}.csv"


def _validate_positive(name: str, value: int) -> int:
    if value < 1:
        raise typer.BadParameter(f"{name} must be >= 1.")
    return value


def _resolve_instance_name(instance_file: Path) -> str:
    return instance_file.stem


def _max_instances_for_single(problem_type: ProblemType, problem_path: Path) -> int:
    if problem_type == ProblemType.KNAPSACK:
        return max(1, len(list(problem_path.glob("*.in"))))

    if problem_type == ProblemType.SPA:
        instances_dir = problem_path / "instances"
        return max(1, len(list(instances_dir.glob("*.txt"))))

    raise typer.BadParameter(f"Unsupported problem type: {problem_type}")


def _load_problem(
    problem_type: ProblemType,
    problem_name: str,
    problem_path: Path,
    max_instances: int,
) -> SimulationProblem:
    if problem_type == ProblemType.KNAPSACK:
        return KnapsackSimulationProblem(name=problem_name, base_path=problem_path, max_instances=max_instances)

    if problem_type == ProblemType.SPA:
        return SPASimulationProblem(name=problem_name, base_path=problem_path, max_instances=max_instances)

    raise typer.BadParameter(f"Unsupported problem type: {problem_type}")


@app.command("run-batch")
def run_batch(
    problem_type: ProblemType = typer.Option(..., "--problem-type", help="Simulation problem type to run."),
    problem_path: Path = typer.Option(..., "--problem-path", help="Path to simulation problem directory."),
    problem_name: str | None = typer.Option(None, "--problem-name", help="Display name for this problem run."),
    max_instances: int = typer.Option(10, "--max-instances", help="Maximum number of instances to load."),
    solution_tool: SolutionTool = typer.Option(
        SolutionTool.PARETOCHAT,
        "--solution-tool",
        help="Which backend solution tool to use.",
    ),
    num_seeds: int = typer.Option(1, "--num-seeds", help="Number of value-function seeds to evaluate."),
    value_function_type: ValueFunctionType = typer.Option(
        ValueFunctionType.ALL,
        "--value-function-type",
        help="Value function type to evaluate.",
    ),
    replications_per_seed: int = typer.Option(
        1,
        "--replications-per-seed",
        help="Replications per seed to account for LLM non-determinism.",
    ),
    max_iterations: int = typer.Option(
        15,
        "--max-iterations",
        help="Maximum number of interactions with the simulated decision maker.",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output-file",
        help="CSV output file path. Defaults to timestamped file in current directory.",
    ),
    export_session_log: bool = typer.Option(
        False,
        "--export-session-log/--no-export-session-log",
        help="If enabled, exports one log.txt per simulation run.",
    ),
    session_artifacts_dir: Path = typer.Option(
        Path("session_artifacts"),
        "--session-artifacts-dir",
        help="Directory where per-run session log artifacts are written.",
    ),
) -> None:
    """Run a simulation batch with CLI-configured settings."""
    if not problem_path.exists():
        raise typer.BadParameter(f"Problem path does not exist: {problem_path}")

    if not problem_path.is_dir():
        raise typer.BadParameter(f"Problem path must be a directory: {problem_path}")

    _validate_positive("max-instances", max_instances)
    _validate_positive("num-seeds", num_seeds)
    _validate_positive("replications-per-seed", replications_per_seed)
    _validate_positive("max-iterations", max_iterations)

    resolved_problem_name = problem_name or _default_problem_name(problem_type, problem_path)
    output_path = output_file or Path(_default_output_file())

    if output_path.parent != Path(".") and not output_path.parent.exists():
        raise typer.BadParameter(f"Output directory does not exist: {output_path.parent}")

    problem = _load_problem(problem_type, resolved_problem_name, problem_path, max_instances)

    run_simulation_batch(
        problem=problem,
        output_file=str(output_path),
        num_seeds=num_seeds,
        replications_per_seed=replications_per_seed,
        max_iterations=max_iterations,
        solution_tool_name=solution_tool.value,
        value_function_type=value_function_type,
        export_session_log=export_session_log,
        session_artifacts_dir=session_artifacts_dir,
    )


@app.command("run_single")
def run_single(
    instance_file: Path = typer.Argument(..., help="Path to the single instance file to run."),
    problem_type: ProblemType = typer.Option(..., "--problem-type", help="Simulation problem type to run."),
    problem_path: Path = typer.Option(..., "--problem-path", help="Path to simulation problem directory."),
    problem_name: str | None = typer.Option(None, "--problem-name", help="Display name for this problem run."),
    solution_tool: SolutionTool = typer.Option(
        SolutionTool.PARETOCHAT,
        "--solution-tool",
        help="Which backend solution tool to use.",
    ),
    num_seeds: int = typer.Option(1, "--num-seeds", help="Number of value-function seeds to evaluate."),
    seed: int | None = typer.Option(None, "--seed", help="Exact random seed value to use for this run."),
    value_function_type: ValueFunctionType = typer.Option(
        ValueFunctionType.ALL,
        "--value-function-type",
        help="Value function type to evaluate.",
    ),
    replications_per_seed: int = typer.Option(
        1,
        "--replications-per-seed",
        help="Replications per seed to account for LLM non-determinism.",
    ),
    max_iterations: int = typer.Option(
        15,
        "--max-iterations",
        help="Maximum number of interactions with the simulated decision maker.",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output-file",
        help="CSV output file path. Defaults to timestamped file in current directory.",
    ),
    export_session_log: bool = typer.Option(
        False,
        "--export-session-log/--no-export-session-log",
        help="If enabled, exports one log.txt per simulation run.",
    ),
    session_artifacts_dir: Path = typer.Option(
        Path("session_artifacts"),
        "--session-artifacts-dir",
        help="Directory where per-run session log artifacts are written.",
    ),
) -> None:
    """Run a simulation batch for exactly one specified instance file."""
    if not problem_path.exists():
        raise typer.BadParameter(f"Problem path does not exist: {problem_path}")

    if not problem_path.is_dir():
        raise typer.BadParameter(f"Problem path must be a directory: {problem_path}")

    if not instance_file.exists():
        raise typer.BadParameter(f"Instance file does not exist: {instance_file}")

    if not instance_file.is_file():
        raise typer.BadParameter(f"Instance file must be a file: {instance_file}")

    _validate_positive("num-seeds", num_seeds)
    _validate_positive("replications-per-seed", replications_per_seed)
    _validate_positive("max-iterations", max_iterations)
    if seed is not None and seed < 0:
        raise typer.BadParameter("seed must be >= 0.")

    resolved_problem_name = problem_name or _default_problem_name(problem_type, problem_path)
    output_path = output_file or Path(_default_output_file())

    if output_path.parent != Path(".") and not output_path.parent.exists():
        raise typer.BadParameter(f"Output directory does not exist: {output_path.parent}")

    max_instances = _max_instances_for_single(problem_type, problem_path)
    problem = _load_problem(problem_type, resolved_problem_name, problem_path, max_instances)

    target_instance_name = _resolve_instance_name(instance_file)
    selected_instance = next((item for item in problem.instances if item.name == target_instance_name), None)
    if selected_instance is None:
        raise typer.BadParameter(
            f"Instance '{target_instance_name}' was not found among loadable instances for {problem_type.value}."
        )

    single_instance_problem = problem.model_copy(update={"instances": [selected_instance]})

    selected_num_seeds = num_seeds
    selected_base_seed = 42
    if seed is not None:
        selected_num_seeds = 1
        selected_base_seed = seed

    run_simulation_batch(
        problem=single_instance_problem,
        output_file=str(output_path),
        num_seeds=selected_num_seeds,
        base_seed=selected_base_seed,
        replications_per_seed=replications_per_seed,
        max_iterations=max_iterations,
        solution_tool_name=solution_tool.value,
        value_function_type=value_function_type,
        export_session_log=export_session_log,
        session_artifacts_dir=session_artifacts_dir,
    )


if __name__ == "__main__":
    app()
