"""Main module to run the biobj discrete choice example."""

import csv
import os
from typing import Annotated

import typer
import yaml

from paretochat.api.cli.interface import get_approval_handler, interactive_main
from paretochat.domain.agents.agent import ParetoChatAgent
from paretochat.domain.algorithm import Algorithm
from paretochat.domain.problem import AlgorithmInfo, ObjectiveSense, ProblemInfo, ScalarizationKind, Solution
from paretochat.domain.scaling import ScalingFunction
from paretochat.domain.scoring import ObjectiveScalarization, ScoringFunction


class Scaler:
    """Base class for scaling values."""

    def scale(self, value: float) -> float:
        raise NotImplementedError

    def descale(self, scaled_value: float) -> float:
        raise NotImplementedError


class ZeroOneScaler(Scaler):
    """Scales values to [0, 1] range."""

    def __init__(self, min_value: float, max_value: float):
        self.min_value = min_value
        self.max_value = max_value

    def scale(self, value: float) -> float:
        if self.max_value > self.min_value:
            return (value - self.min_value) / (self.max_value - self.min_value)
        else:
            return 0.0

    def descale(self, scaled_value: float) -> float:
        return scaled_value * (self.max_value - self.min_value) + self.min_value


class ZeroOneScalingFunction(ScalingFunction):
    """Scales values to [0, 1] range."""

    def __init__(self, scalers: dict[str, Scaler]):
        self.scalers = scalers

    def fit(self):
        """Fit the scaling function to the data."""
        pass

    def transform(self, objective_values: dict[str, float]) -> dict[str, float]:
        return {k: self.scalers[k].scale(v) for k, v in objective_values.items()}

    def inverse_transform(self, objective_values: dict[str, float]) -> dict[str, float]:
        return {k: self.scalers[k].descale(v) for k, v in objective_values.items()}


class BiObjDiscreteChoiceAlgorithm(Algorithm):
    """A class representing the biobj discrete choice algorithm."""

    def __init__(self):
        """Initialize the biobj discrete choice algorithm."""
        super().__init__()
        self._problem_info = self._load_problem_info()
        self._algorithm_info = self._create_algorithm_info()
        self._solutions = self._load_solutions()
        self._scalers = self._create_scalers()
        self._scoring_function = self._create_scoring_function()

    def _load_problem_info(self) -> ProblemInfo:
        """Load problem information from info.yaml."""
        with open(os.path.join(os.path.dirname(__file__), "info.yaml"), "r") as f:
            info = yaml.safe_load(f)

        return ProblemInfo.model_validate(info.get("problem", {}))

    def _load_solutions(self) -> list[Solution]:
        """Load the solutions from a csv file."""
        solutions = []
        with open(os.path.join(os.path.dirname(__file__), "solutions.csv"), "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip the header row
            for row in reader:
                solution = Solution(
                    objective_values={
                        "delivery_time": float(row[0]),
                        "cost": float(row[1]),
                    },
                )
                solutions.append(solution)
        return solutions

    def _create_scalers(self) -> dict[str, Scaler]:
        """Create a scaler for the solutions."""
        delivery_time_values = [s.objective_values["delivery_time"] for s in self._solutions]
        cost_values = [s.objective_values["cost"] for s in self._solutions]
        min_delivery_time, max_delivery_time = min(delivery_time_values), max(delivery_time_values)
        min_cost, max_cost = min(cost_values), max(cost_values)
        return {
            "delivery_time": ZeroOneScaler(min_delivery_time, max_delivery_time),
            "cost": ZeroOneScaler(min_cost, max_cost),
        }

    def _create_algorithm_info(self) -> AlgorithmInfo:
        """Return the algorithm info."""
        return AlgorithmInfo(
            name="biobj_discrete_choice",
            description="A biobj discrete choice algorithm.",
            optimization_direction=ObjectiveSense.MINIMIZE,
            supported_scalarizations=[
                ScalarizationKind.WEIGHTED_SUM,
                ScalarizationKind.REFERENCE_POINT,
                ScalarizationKind.LEXICOGRAPHIC,
            ],
            supports_obj_bounds=True,
        )

    def _create_scoring_function(self) -> ScoringFunction:
        """Create a scoring function."""
        return ScoringFunction(
            problem_info=self._problem_info,
            algorithm_info=self._algorithm_info,
            scaling_function=ZeroOneScalingFunction(self._scalers),
        )

    @property
    def algorithm_info(self) -> AlgorithmInfo:
        """Return the algorithm info."""
        return self._algorithm_info

    def run(self, objective_scalarization: ObjectiveScalarization) -> Solution | None:
        """Run the biobj discrete choice algorithm."""
        # filter the solutions if there are bounds on the objectives
        filtered_solutions = self._solutions
        for objective_name, (min_value, max_value) in objective_scalarization.get_bounds_dict().items():
            if max_value is not None:
                filtered_solutions = [s for s in filtered_solutions if s.objective_values[objective_name] <= max_value]
            if min_value is not None:
                filtered_solutions = [s for s in filtered_solutions if s.objective_values[objective_name] >= min_value]

        if not filtered_solutions:
            return None

        # score the solutions
        scores = [self._scoring_function.score(objective_scalarization, s.objective_values) for s in filtered_solutions]

        # return the best solution
        if self._algorithm_info.optimization_direction == ObjectiveSense.MINIMIZE:
            return filtered_solutions[scores.index(min(scores))]
        else:
            return filtered_solutions[scores.index(max(scores))]


def main(
    auto_approve: Annotated[
        bool,
        typer.Option(help="Run the algorithm without asking for approval. Use --no-auto-approve to confirm each run."),
    ] = True,
):
    """Main function to run the biobj discrete choice example."""
    domain_info_path = os.path.join(os.path.dirname(__file__), "info.yaml")
    algorithm = BiObjDiscreteChoiceAlgorithm()
    agent = ParetoChatAgent(domain_info_path, algorithm, approval_handler=get_approval_handler(auto_approve))
    example_queries = [
        "Find a solution with delivery time under 45 minutes.",
        "Find a solution with cost under 20.",
        "Find a solution that balances delivery time and cost.",
        "Prioritize delivery time over cost.",
        "Prioritize cost over delivery time.",
    ]
    interactive_main(agent, example_queries)


if __name__ == "__main__":
    typer.run(main)
