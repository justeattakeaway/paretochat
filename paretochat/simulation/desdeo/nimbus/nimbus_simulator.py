from typing import Any, Optional

from desdeo.mcdm.nimbus import (
    generate_starting_point,
    infer_classifications,
    solve_sub_problems,
)
from desdeo.mcdm.reference_point_method import rpm_intermediate_solutions
from desdeo.problem import Problem as DESDEOProblem
from desdeo.tools import payoff_table_method
from desdeo.tools.generics import SolverResults
import numpy as np
from pydantic import BaseModel, Field

from paretochat.domain.problem import Solution


class NimbusInput(BaseModel):
    """Structured NIMBUS input suitable for wrapping as an AI tool request."""

    classifications: Optional[dict[str, tuple[str, Optional[float]]]] = Field(
        default=None,
        description=(
            "Classification of objectives as dict mapping objective symbol to a tuple of (class, level). "
            "Classes: '<' (improve), '<=' (improve until level), '=' (keep), '>=' (worse until level), '0' (free). "
            "Level is None for '<', '=', '0' and numeric for '<=' and '>='. "
            "Omit on the first call to request a starting point."
        ),
    )
    current_objectives: Optional[dict[str, float]] = Field(
        default=None,
        description="Current objective values that the classifications are based on. Omit on the first call.",
    )


class NimbusOutput(BaseModel):
    """Structured NIMBUS output suitable for wrapping as an AI tool response."""

    generated_solutions: list[Solution] = Field(
        default_factory=list,
        description="The four solutions generated in this iteration (from 4 different scalarization functions).",
    )


class NimbusSession(BaseModel):
    """In-memory NIMBUS session state."""

    objective_symbols: list[str]
    ideal: dict[str, float]
    nadir: dict[str, float]
    problem: DESDEOProblem = Field(description="The DESDEO problem instance being solved in this session.")
    step_number: int = 0
    archived_solutions: list[Solution] = Field(
        default_factory=list,
        description="Solutions archived by the DM for later review.",
    )


class NimbusMethodSimulator:
    """Stateful DESDEO NIMBUS interface designed for AIDM tool integration."""

    def __init__(
        self,
        desdeo_problem: DESDEOProblem,
    ):
        """Initialize the NIMBUS simulator with a DESDEO problem.

        Args:
            desdeo_problem: A pre-built DESDEO Problem instance.
        """
        self._desdeo_problem = desdeo_problem
        self._session: NimbusSession | None = None

    @property
    def has_active_session(self) -> bool:
        return self._session is not None

    def start_session(self) -> None:
        """Initialize a new NIMBUS session."""
        if self._desdeo_problem is not None:
            problem = self._desdeo_problem
        else:
            raise ValueError("DESDEO problem must be provided.")

        ideal, nadir = payoff_table_method(problem)
        problem = problem.update_ideal_and_nadir(
            new_ideal=ideal,
            new_nadir=nadir,
        )
        objective_symbols = [objective.symbol for objective in problem.objectives]
        self._session = NimbusSession(
            objective_symbols=objective_symbols,
            ideal={symbol: float(ideal[symbol]) for symbol in objective_symbols},
            nadir={symbol: float(nadir[symbol]) for symbol in objective_symbols},
            problem=problem,
            step_number=0,
            archived_solutions=[],
        )

    def generate_starting_point(
        self,
        reference_point: Optional[dict[str, float]] = None,
    ) -> Solution:
        """Generate a starting Pareto optimal solution for NIMBUS iteration.

        If no reference point is given, uses the compromise reference point
        (midpoint between ideal and nadir for each objective).

        Args:
            reference_point: Optional reference point dict keyed by objective symbols.
                           If None, uses compromise point.

        Returns:
            A Solution object representing the starting point.

        Raises:
            ValueError: If session is not active or reference point is invalid.
        """
        if not self.has_active_session:
            self.start_session()

        # Build compromise reference point if not provided
        if reference_point is None:
            reference_point = {
                symbol: self._session.nadir[symbol] + (self._session.ideal[symbol] - self._session.nadir[symbol]) / 2.0
                for symbol in self._session.objective_symbols
            }

        # Ensure all objectives are in the reference point
        missing = [s for s in self._session.objective_symbols if s not in reference_point]
        if missing:
            raise ValueError(f"Missing reference point values for objectives: {', '.join(missing)}")

        # Generate starting point using NIMBUS's generate_starting_point
        result = generate_starting_point(
            problem=self._session.problem,
            reference_point=reference_point,
            solver_options={"LogToConsole": 0, "OutputFlag": 0},
        )

        return self._extract_solution(result.optimal_objectives, self._session.objective_symbols)

    def find_new_solutions(
        self,
        input: NimbusInput,
    ) -> NimbusOutput:
        """Process DM classifications and generate 4 new Pareto optimal solutions using NIMBUS.

        The 4 solutions come from applying 4 different scalarization functions:
        NIMBUS, STOM, ASF (achievement scalarizing function), and GUESS.

        Args:
            input: NimbusInput with classifications and current objectives.

        Returns:
            NimbusOutput with 4 generated solutions.

        Raises:
            ValueError: If classifications are invalid (e.g., missing objectives).
            Exception: If DESDEO raises NimbusError due to invalid classification combination.
        """
        if not self.has_active_session:
            self.start_session()

        if input.current_objectives is None:
            starting_point = self.generate_starting_point()
            return NimbusOutput(generated_solutions=[starting_point])

        if input.classifications is None:
            raise ValueError("classifications must be provided after the starting point has been generated.")

        # Validate and convert classifications to reference_point dict
        reference_point = self._classifications_to_reference_point(
            input.classifications,
            input.current_objectives,
        )

        # Solve 4 sub-problems using different scalarization functions
        raw_results = solve_sub_problems(
            problem=self._session.problem,
            current_objectives=input.current_objectives,
            reference_point=reference_point,
            num_desired=4,
            solver_options={"LogToConsole": 0, "OutputFlag": 0},
        )

        solutions = self._extract_solutions(raw_results, self._session.objective_symbols)
        self._session.step_number += 1

        return NimbusOutput(generated_solutions=solutions)

    def infer_classifications(
        self,
        current_objectives: dict[str, float],
        reference_point: dict[str, float],
    ) -> dict[str, tuple[str, Optional[float]]]:
        """Validate and infer classifications from objective values and reference point.

        Useful for the AI agent to double-check that classifications are valid before
        calling find_new_solutions.

        Args:
            current_objectives: Current objective values.
            reference_point: Reference point representing DM's classifications.

        Returns:
            Dict mapping objective symbols to (class, level) tuples.

        Raises:
            ValueError: If objectives or reference point are missing required keys.
        """
        if not self.has_active_session:
            self.start_session()

        missing_obj = [s for s in self._session.objective_symbols if s not in current_objectives]
        missing_rp = [s for s in self._session.objective_symbols if s not in reference_point]

        if missing_obj:
            raise ValueError(f"Missing current_objectives for symbols: {', '.join(missing_obj)}")
        if missing_rp:
            raise ValueError(f"Missing reference_point for symbols: {', '.join(missing_rp)}")

        # Infer classifications using DESDEO's function
        classifications = infer_classifications(
            problem=self._session.problem,
            current_objectives=current_objectives,
            reference_point=reference_point,
        )

        return classifications

    def generate_intermediate_solutions(
        self,
        solution_1: Solution,
        solution_2: Solution,
        num_desired: int,
    ) -> list[Solution]:
        """Generate intermediate solutions between two given solutions.

        Uses rpm_intermediate_solutions to generate solutions in objective space
        (recommended by DESDEO authors over decision space approach).

        Args:
            solution_1: First solution.
            solution_2: Second solution.
            num_desired: Number of intermediate solutions to generate.

        Returns:
            List of intermediate Solution objects.

        Raises:
            ValueError: If session is not active or inputs are invalid.
        """
        if not self.has_active_session:
            self.start_session()

        if num_desired < 1:
            raise ValueError("num_desired must be at least 1")

        # Use rpm_intermediate_solutions as recommended by DESDEO
        raw_results = rpm_intermediate_solutions(
            problem=self._session.problem,
            solution_1=solution_1.objective_values,
            solution_2=solution_2.objective_values,
            num_desired=num_desired,
            solver_options={"LogToConsole": 0, "OutputFlag": 0},
        )

        return self._extract_solutions(raw_results, self._session.objective_symbols)

    def archive_solution(self, solution: Solution) -> None:
        """Archive a solution for later review.

        Args:
            solution: Solution to archive.

        Raises:
            ValueError: If session is not active.
        """
        if not self.has_active_session:
            self.start_session()

        self._session.archived_solutions.append(solution)

    def get_archived_solutions(self) -> list[Solution]:
        """Get all archived solutions.

        Returns:
            List of archived solutions.

        Raises:
            ValueError: If session is not active.
        """
        if not self.has_active_session:
            self.start_session()

        return self._session.archived_solutions.copy()

    def reset(self) -> None:
        """Clear active session state."""
        self._session = None

    def _classifications_to_reference_point(
        self,
        classifications: dict[str, tuple[str, Optional[float]]],
        current_objectives: dict[str, float],
    ) -> dict[str, float]:
        """Convert NIMBUS classifications to a reference_point dict for solve_sub_problems.

        The conversion maps NIMBUS class notation to reference point values:
        - '<': ideal value
        - '<=': aspiration level (provided value)
        - '=': current value (from classification tuple, if present)
        - '>=': reservation level (provided value)
        - '0': nadir value

        Args:
            classifications: Dict mapping objective symbols to (class, level) tuples.
            current_objectives: Current objective values for the classified solution.

        Returns:
            Dict mapping objective symbols to reference point values.

        Raises:
            ValueError: If classifications dict is missing any required objective.
        """
        if not self.has_active_session:
            self.start_session()

        objective_symbols = self._session.objective_symbols
        ideal = self._session.ideal
        nadir = self._session.nadir

        missing = [s for s in objective_symbols if s not in classifications]
        if missing:
            raise ValueError(f"Classifications missing entries for objectives: {', '.join(missing)}")

        missing_current = [s for s in objective_symbols if s not in current_objectives]
        if missing_current:
            raise ValueError(f"current_objectives missing entries for objectives: {', '.join(missing_current)}")

        improving = False
        worsening = False
        reference_point: dict[str, float] = {}

        for symbol in objective_symbols:
            class_type, level = classifications[symbol]
            current_value = float(current_objectives[symbol])
            ideal_value = float(ideal[symbol])
            maximize = self._session.problem.get_objective(symbol).maximize

            if class_type == "<":
                reference_point[symbol] = ideal_value
                improving = True
            elif class_type == "<=":
                if level is None:
                    raise ValueError(f"Classification '<=' for {symbol} requires an aspiration level.")
                aspiration_value = float(level)
                if (maximize and aspiration_value <= current_value) or (
                    not maximize and aspiration_value >= current_value
                ):
                    raise ValueError("Reference point not better than current value for classification <=")
                reference_point[symbol] = aspiration_value
                improving = True
            elif class_type == "=":
                reference_point[symbol] = current_value
            elif class_type == ">=":
                if level is None:
                    raise ValueError(f"Classification '>=' for {symbol} requires a reservation level.")
                reservation_value = float(level)
                if (maximize and reservation_value >= current_value) or (
                    not maximize and reservation_value <= current_value
                ):
                    raise ValueError("Reference point not worse than current value for classification >=")
                reference_point[symbol] = reservation_value
                worsening = True
            elif class_type == "0":
                reference_point[symbol] = float(nadir[symbol])
                worsening = True
            else:
                raise ValueError(f"Unsupported NIMBUS classification '{class_type}' for {symbol}.")

        if not improving or not worsening:
            raise ValueError(
                "NIMBUS classifications must include at least one of '<'/'<=' and at least one of '>='/'0'."
            )

        return reference_point

    @staticmethod
    def _extract_solution(raw_solution: Any, objective_symbols: list[str]) -> Solution | None:
        if raw_solution is None:
            return None

        if isinstance(raw_solution, dict):
            values = {symbol: float(raw_solution[symbol]) for symbol in objective_symbols if symbol in raw_solution}
            if values:
                return Solution(objective_values=values)
            return None

        raw_array = np.asarray(raw_solution, dtype=float).flatten()
        if raw_array.size < len(objective_symbols):
            return None

        return Solution(
            objective_values={symbol: float(raw_array[idx]) for idx, symbol in enumerate(objective_symbols)}
        )

    def _extract_solutions(
        self,
        raw_results: list[SolverResults],
        objective_symbols: list[str],
    ) -> list[Solution]:
        solutions: list[Solution] = []
        for result in raw_results:
            raw_objectives = getattr(result, "optimal_objectives", None)
            solution = self._extract_solution(raw_objectives, objective_symbols)
            if solution is not None:
                solutions.append(solution)
        return solutions


if __name__ == "__main__":
    from paretochat.simulation.desdeo.desdeo_problem import create_knapsack_problem
    from paretochat.simulation.knapsack import KnapsackSimulationProblem

    simulation_problem = KnapsackSimulationProblem(
        name="2OKP-Random",
        base_path="paretochat/simulation/instances/mobkp_instances/random/2D",
        max_instances=1,
    )
    instance = simulation_problem.instances[0]
    desdeo_problem = create_knapsack_problem(simulation_problem, instance)
    simulator = NimbusMethodSimulator(desdeo_problem=desdeo_problem)

    # Generate starting point
    print("=== Generating Starting Point ===")
    starting_point = simulator.generate_starting_point(reference_point=None)
    print(f"Starting point objectives: {starting_point.objective_values}")

    # Define classifications for first iteration
    print("\n=== First NIMBUS Iteration ===")
    classifications = {
        "Objective_1": ("<", None),
        "Objective_2": (">=", 11000.0),
    }
    input_data = NimbusInput(
        classifications=classifications,
        current_objectives=starting_point.objective_values,
    )

    output = simulator.find_new_solutions(input_data)
    print(f"Generated {len(output.generated_solutions)} solutions:")
    for idx, sol in enumerate(output.generated_solutions):
        print(f"  Solution {idx + 1}: {sol.objective_values}")

    # Generate intermediate solutions between first two generated solutions
    if len(output.generated_solutions) >= 2:
        print("\n=== Generating Intermediate Solutions ===")
        sol1 = output.generated_solutions[0]
        sol2 = output.generated_solutions[1]
        intermediate_solutions = simulator.generate_intermediate_solutions(
            solution_1=sol1,
            solution_2=sol2,
            num_desired=3,
        )
        print(f"Generated {len(intermediate_solutions)} intermediate solutions:")
        for idx, sol in enumerate(intermediate_solutions):
            print(f"  Intermediate Solution {idx + 1}: {sol.objective_values}")
