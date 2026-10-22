from typing import Any, Optional

from desdeo.mcdm.reference_point_method import rpm_solve_solutions
from desdeo.problem import Problem as DESDEOProblem
from desdeo.tools import payoff_table_method
from desdeo.tools.generics import SolverResults
import numpy as np
from pydantic import BaseModel, Field

from paretochat.domain.problem import Solution
from paretochat.simulation.desdeo.desdeo_problem import create_knapsack_problem
from paretochat.simulation.knapsack import KnapsackSimulationProblem


class RPMInput(BaseModel):
    """Structured RPM input suitable for wrapping as an AI tool request."""

    reference_point: Optional[dict[str, float]] = Field(
        None,
        description="The reference point submitted by the DM for this iteration, if applicable.",
    )


class RMPOutput(BaseModel):
    """Structured RPM output suitable for wrapping as an AI tool response."""

    generated_solutions: list[Solution] = Field(
        default_factory=list,
        description="The list of solutions generated in this iteration.",
    )


class RPMSession(BaseModel):
    """In-memory RPM session state."""

    objective_symbols: list[str]
    ideal: dict[str, float]
    nadir: dict[str, float]
    problem: DESDEOProblem = Field(description="The DESDEO problem instance being solved in this session.")
    step_number: int = 0


class ReferencePointMethodSimulator:
    """Stateful DESDEO RPM interface designed for future AIDM tool integration."""

    def __init__(
        self,
        desdeo_problem: DESDEOProblem,
    ):
        """Initialize the RPM simulator with a DESDEO problem.

        Args:
            desdeo_problem: A pre-built DESDEO Problem instance.

        """
        self._desdeo_problem = desdeo_problem
        self._session: RPMSession | None = None

    @property
    def has_active_session(self) -> bool:
        return self._session is not None

    def start_session(self) -> None:
        """Initialize a new RPM session and return the first request payload."""
        # Build or retrieve the DESDEO problem
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
        self._session = RPMSession(
            objective_symbols=objective_symbols,
            ideal={symbol: float(ideal[symbol]) for symbol in objective_symbols},
            nadir={symbol: float(nadir[symbol]) for symbol in objective_symbols},
            problem=problem,
            step_number=0,
        )

    def find_new_solutions(
        self,
        input: RPMInput,
    ) -> RMPOutput:
        """Process a DM-submitted reference point, compute new solutions using the RPM, and return the results."""
        if not self.has_active_session:
            self.start_session()

        reference_point = input.reference_point if input.reference_point is not None else self._session.ideal
        rp_vec = self._normalize_reference_point(reference_point, self._session.objective_symbols)
        rp_dict = {symbol: float(rp_vec[idx]) for idx, symbol in enumerate(self._session.objective_symbols)}

        raw_results = rpm_solve_solutions(
            problem=self._session.problem, reference_point=rp_dict, solver_options={"LogToConsole": 0, "OutputFlag": 0}
        )
        solutions = self._extract_solutions(raw_results, self._session.objective_symbols)

        self._session.step_number += 1

        return RMPOutput(
            generated_solutions=solutions,
        )

    def reset(self) -> None:
        """Clear active session state."""
        self._session = None

    @staticmethod
    def _normalize_reference_point(
        reference_point: dict[str, float] | np.ndarray,
        objective_symbols: list[str],
    ) -> np.ndarray:
        if isinstance(reference_point, dict):
            missing = [symbol for symbol in objective_symbols if symbol not in reference_point]
            if missing:
                missing_str = ", ".join(missing)
                raise ValueError(f"Missing reference point values for objectives: {missing_str}")

            return np.array([reference_point[symbol] for symbol in objective_symbols], dtype=float)

        rp_vec = np.array(reference_point, dtype=float)
        if rp_vec.shape != (len(objective_symbols),):
            raise ValueError(f"Reference point must have shape ({len(objective_symbols)},), got {rp_vec.shape}.")

        return rp_vec

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
    simulation_problem = KnapsackSimulationProblem(
        name="2OKP-Random",
        base_path="paretochat/simulation/instances/mobkp_instances/random/2D",
        max_instances=1,
    )
    instance = simulation_problem.instances[0]
    desdeo_problem = create_knapsack_problem(simulation_problem, instance)
    simulator = ReferencePointMethodSimulator(desdeo_problem=desdeo_problem)
    reference_point = None
    for i in range(3):
        print(f"\n=== RPM Iteration {i + 1} ===")
        print(f"Submitting reference point: {reference_point}")
        solns = simulator.find_new_solutions(RPMInput(reference_point=reference_point)).generated_solutions
        print(f"Iteration {i + 1} generated {len(solns)} solutions:")
        for idx, sol in enumerate(solns):
            print(f"  Solution {idx + 1} objective values: {sol.objective_values}")
        # take the mean of the generated solutions as the next reference point for demonstration purposes
        if solns:
            reference_point = {
                symbol: np.mean([sol.objective_values[symbol] for sol in solns]).item()
                for symbol in simulator._session.objective_symbols
            }
        else:
            reference_point = None
