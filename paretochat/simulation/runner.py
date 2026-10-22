import csv
import json
from pathlib import Path
import re
import time
import traceback
from typing import Dict

from paretochat.domain.agents.agent import ParetoChatAgent
from paretochat.domain.problem import ObjectiveSense
from paretochat.domain.solution_pool import get_solutions, reset_solution_pool
from paretochat.simulation.ai_dm import AIDecisionMaker
from paretochat.simulation.base import SolutionTool, ValueFunctionType
from paretochat.simulation.desdeo.desdeo_problem import create_knapsack_problem, create_spa_problem
from paretochat.simulation.desdeo.nimbus.nimbus_simulator import NimbusMethodSimulator
from paretochat.simulation.desdeo.nimbus.tool import create_desdeo_nimbus_tool
from paretochat.simulation.desdeo.rpm.rpm_simulator import ReferencePointMethodSimulator
from paretochat.simulation.desdeo.rpm.tool import create_desdeo_rpm_tool
from paretochat.simulation.evaluator import PerformanceEvaluator
from paretochat.simulation.knapsack import KnapsackSimulationProblem, KnapsackSolver
from paretochat.simulation.paretochat.tool import create_paretochat_tool
from paretochat.simulation.simulation_problem import SimulationProblem, SimulationProblemInstance
from paretochat.simulation.spa import SPASimulationProblem, SPASolver
from paretochat.simulation.value_function import ValueFunction, ValueFunctionFactory


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip("-_.")
    return cleaned or "na"


def _write_session_artifacts(
    artifacts_dir: Path,
    session_messages: list[str],
    export_solution_pool: bool,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "log.txt").write_text("\n\n".join(session_messages), encoding="utf-8")

    if not export_solution_pool:
        return

    solution_rows = []
    for item in get_solutions():
        feedback = item.feedback
        solution_rows.append(
            {
                "trial_number": item.trial_number,
                "user_input": item.user_input,
                "duplicate": item.duplicate,
                "scalarization_kind": item.objective_scalarization.kind,
                "scalarization_parameters": json.dumps(item.objective_scalarization.get_params_dict(), sort_keys=True),
                "scalarization_bounds": json.dumps(item.objective_scalarization.get_bounds_dict(), sort_keys=True),
                "has_solution": item.solution is not None,
                "solution_objective_values": (
                    json.dumps(item.solution.objective_values, sort_keys=True) if item.solution is not None else ""
                ),
                "feedback_sentiment": feedback.sentiment if feedback is not None else "",
                "feedback_text": feedback.feedback_text if feedback is not None else "",
                "feedback_save_last_solution": feedback.save_last_solution if feedback is not None else "",
            }
        )

    with open(artifacts_dir / "solution_pool.csv", "w", newline="") as f:
        fieldnames = [
            "trial_number",
            "user_input",
            "duplicate",
            "scalarization_kind",
            "scalarization_parameters",
            "scalarization_bounds",
            "has_solution",
            "solution_objective_values",
            "feedback_sentiment",
            "feedback_text",
            "feedback_save_last_solution",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(solution_rows)


def run_single_simulation(
    problem: SimulationProblem,
    instance: SimulationProblemInstance,
    value_function: ValueFunction,
    run_id: int,
    seed: int,
    replication_index: int = 1,
    max_iterations: int = 15,
    solution_tool_name: SolutionTool = SolutionTool.PARETOCHAT,
    export_session_log: bool = False,
    session_artifacts_dir: str | Path = "session_artifacts",
    maximize: bool = True,
) -> Dict:
    """Run a single simulation instance."""

    reset_solution_pool()

    # 1. Setup Solution Tool
    if solution_tool_name == SolutionTool.PARETOCHAT:
        if isinstance(problem, KnapsackSimulationProblem):
            algorithm = KnapsackSolver(problem.domain_info, instance.input_data)
        elif isinstance(problem, SPASimulationProblem):
            algorithm = SPASolver(problem.domain_info, instance.input_data)
        else:
            raise ValueError(f"Unsupported simulation problem type for paretochat tool: {type(problem).__name__}")
        paretochat_agent = ParetoChatAgent(problem.domain_info_path, algorithm)
        solution_tool = create_paretochat_tool(paretochat_agent)
    elif solution_tool_name == SolutionTool.DESDEO_RPM:
        if isinstance(problem, KnapsackSimulationProblem):
            desdeo_problem = create_knapsack_problem(problem, instance)
        elif isinstance(problem, SPASimulationProblem):
            desdeo_problem = create_spa_problem(problem, instance)
        else:
            raise ValueError(f"Unsupported simulation problem type for desdeo tool: {type(problem).__name__}")
        rpm_simulator = ReferencePointMethodSimulator(desdeo_problem=desdeo_problem)
        solution_tool = create_desdeo_rpm_tool(rpm_simulator)
    elif solution_tool_name == SolutionTool.DESDEO_NIMBUS:
        if isinstance(problem, KnapsackSimulationProblem):
            desdeo_problem = create_knapsack_problem(problem, instance)
        elif isinstance(problem, SPASimulationProblem):
            desdeo_problem = create_spa_problem(problem, instance)
        else:
            raise ValueError(f"Unsupported simulation problem type for desdeo tool: {type(problem).__name__}")
        nimbus_simulator = NimbusMethodSimulator(desdeo_problem=desdeo_problem)
        solution_tool = create_desdeo_nimbus_tool(nimbus_simulator)
    else:
        raise ValueError(f"Unknown solution tool: {solution_tool_name}")

    # 2. Setup AI Decision Maker
    # We pass the evaluate method of the value function
    ai_dm = AIDecisionMaker(
        problem_info=problem.domain_info.problem,
        preference_function=value_function.evaluate,
    )

    # 3. Run Simulation
    start_time = time.time()
    print(f"--- Starting Run {run_id} ---")
    print(f"Value Function: {value_function.type}")
    print(f"Weights: {getattr(value_function, 'weights', 'N/A')}")

    simulation_result = ai_dm.run(solution_tool=solution_tool, max_iterations=max_iterations)
    duration = time.time() - start_time

    # 4. Evaluate Performance
    evaluator = PerformanceEvaluator(instance)
    metrics = evaluator.evaluate(simulation_result, value_function, maximize=maximize)

    # 5. Format Result
    result_record = {
        "run_id": run_id,
        "problem": problem.name,
        "instance": instance.name,
        "pareto_size": metrics.pareto_size,
        "solution_tool": solution_tool_name,
        "vf_type": value_function.type,
        "vf_weights": str(getattr(value_function, "weights", {})),
        "duration_seconds": round(duration, 2),
        "ai_dm_total_time_seconds": round(simulation_result.total_dm_time_seconds, 4),
        "solver_time_seconds": round(simulation_result.solver_time_seconds, 4),
        "ai_dm_non_solver_time_seconds": round(simulation_result.non_solver_time_seconds, 4),
        "solver_call_count": simulation_result.solver_call_count,
        "solution_found": metrics.solution_found,
        "num_eval_sols": simulation_result.num_eval_sols,
        "best_found_value": metrics.found_value,
        "found_value_rank": metrics.found_value_rank,
        "found_value_rank_normalized": metrics.found_value_rank_normalized,
        "best_possible_value": metrics.best_possible_value,
        "value_gap": metrics.value_gap,
        "value_ratio": metrics.value_ratio,
        "is_optimal": metrics.is_optimal,
        "solution_history": simulation_result.history,
    }

    print(f"Run {run_id} Completed.")
    if metrics.solution_found:
        print(f"Gap: {metrics.value_gap:.4f}, Ratio: {metrics.value_ratio:.4f}")
    else:
        print("No solution found.")

    if export_session_log:
        artifact_stem = (
            f"run_{run_id:04d}"
            f"__problem_{_slugify(problem.name)}"
            f"__instance_{_slugify(instance.name)}"
            f"__tool_{_slugify(str(solution_tool_name))}"
            f"__vf_{_slugify(str(value_function.type))}"
            f"__seed_{seed}"
            f"__rep_{replication_index}"
        )
        artifact_path = Path(session_artifacts_dir) / artifact_stem
        _write_session_artifacts(
            artifacts_dir=artifact_path,
            session_messages=simulation_result.session_messages,
            export_solution_pool=(solution_tool_name == SolutionTool.PARETOCHAT),
        )

    return result_record


def run_simulation_batch(
    problem: SimulationProblem,
    output_file: str = "simulation_results.csv",
    num_seeds: int = 1,
    base_seed: int = 42,
    replications_per_seed: int = 1,
    max_iterations: int = 15,
    solution_tool_name: SolutionTool = SolutionTool.PARETOCHAT,
    value_function_type: ValueFunctionType = ValueFunctionType.ALL,
    export_session_log: bool = False,
    session_artifacts_dir: str | Path = "session_artifacts",
):
    """Run a batch of simulations."""

    # Initialize Problem (Knapsack)
    try:
        print(f"Loaded problem: {problem.name}")
    except Exception as e:
        print(f"Failed to load problem: {e}")
        return

    results = []
    run_counter = 0
    objective_names = [obj.name for obj in problem.domain_info.problem.objectives]
    objective_senses = {obj.name: obj.sense for obj in problem.domain_info.problem.objectives}

    maximize = all(obj.sense == ObjectiveSense.MAXIMIZE for obj in problem.domain_info.problem.objectives)
    minimize = all(obj.sense == ObjectiveSense.MINIMIZE for obj in problem.domain_info.problem.objectives)
    if not (maximize or minimize):
        raise ValueError("Mixed objective senses are not supported in this simulation setup.")

    if value_function_type == ValueFunctionType.ALL:
        vf_types = [
            ValueFunctionType.LINEAR,
            ValueFunctionType.NONLINEAR,
            ValueFunctionType.SUM_SQUARED_DISTANCE,
        ]
    else:
        vf_types = [value_function_type]

    for instance in problem.instances:
        for vf_type in vf_types:
            for seed_idx in range(num_seeds):
                # Generate random value function
                current_seed = base_seed + seed_idx  # Deterministic seed generation

                if vf_type == ValueFunctionType.LINEAR:
                    vf = ValueFunctionFactory.create_random_linear(
                        objective_names, maximize=maximize, seed=current_seed
                    )
                elif vf_type == ValueFunctionType.NONLINEAR:
                    vf = ValueFunctionFactory.create_random_nonlinear(
                        objective_names, maximize=maximize, seed=current_seed
                    )
                elif vf_type == ValueFunctionType.SUM_SQUARED_DISTANCE:
                    vf = ValueFunctionFactory.create_random_weigted_squared_sums_distance_to_ideal_point(
                        objective_names=objective_names,
                        objective_senses=objective_senses,
                        pareto_front=instance.pareto_front,
                        seed=current_seed,
                    )
                else:
                    raise ValueError(f"Unknown value function type: {vf_type}")

                # Run replications for this VF to account for LLM non-determinism
                for replication_idx in range(replications_per_seed):
                    run_counter += 1
                    try:
                        record = run_single_simulation(
                            problem,
                            instance,
                            vf,
                            run_counter,
                            seed=current_seed,
                            replication_index=replication_idx + 1,
                            max_iterations=max_iterations,
                            solution_tool_name=solution_tool_name,
                            export_session_log=export_session_log,
                            session_artifacts_dir=session_artifacts_dir,
                            maximize=maximize,
                        )
                        results.append(record)
                    except Exception as e:
                        print(f"Run {run_counter} failed: {e}")
                        traceback.print_exc()

    # Save Results
    if results:
        fieldnames = list(results[0].keys())
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSimulation complete. Results saved to {output_file}")
    else:
        print("\nNo results to save.")
