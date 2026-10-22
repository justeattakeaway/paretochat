"""A module defining the SolutionPool class and related tools."""

from typing import Optional

from pydantic import BaseModel, Field

from paretochat.domain.problem import Solution
from paretochat.domain.scoring import ObjectiveScalarization
from paretochat.domain.user import UserFeedback


class SolutionWithHistory(BaseModel):
    """A class representing the history of a solution."""

    trial_number: Optional[int] = Field(
        default=None,
        description="The sequence number of the trial to find a solution (if any).",
    )
    user_input: str = Field(
        description="The user input that led to the solution.",
    )
    objective_scalarization: ObjectiveScalarization = Field(
        description="The objective scalarization that led to the solution.",
    )
    solution: Optional[Solution] = Field(
        default=None,
        description="The solution (if found).",
    )
    duplicate: bool = Field(
        default=False,
        description="Whether the solution is a duplicate of an existing solution.",
    )
    feedback: Optional[UserFeedback] = Field(
        default=None,
        description="The feedback on the solution (if any).",
    )


class SolutionPool(BaseModel):
    """A class representing a pool of solutions."""

    solutions: list[SolutionWithHistory] = Field(
        default_factory=list,
        description="List of solutions in the pool.",
    )

    def add_solution(self, solution: SolutionWithHistory) -> None:
        """Add a new solution to the pool."""
        solution.trial_number = len(self.solutions) + 1
        is_duplicate = self.is_duplicate_solution(solution.solution) if solution.solution is not None else False
        solution.duplicate = is_duplicate
        self.solutions.append(solution)

    def add_feedback(self, feedback: UserFeedback) -> None:
        """Add feedback to the last solution."""
        if self.solutions:
            self.solutions[-1].feedback = feedback

    def is_duplicate_solution(self, solution: Solution) -> bool:
        """Check if the solution is a duplicate of any existing solution in the pool.

        Returns True if the solution is a duplicate, False otherwise.
        """
        for s in self.solutions:
            if s.solution is None:
                continue

            # Compare objective values to determine if the solution is a duplicate
            if s.solution.obj_equals(solution):
                return True

        return False

    def get_duplicate_solutions(self, solution: Solution) -> list[SolutionWithHistory]:
        """Retrieve all duplicate solutions for a given solution."""
        return [s for s in self.solutions if s.solution and s.solution.obj_equals(solution)]

    def get_solutions(self) -> list[SolutionWithHistory]:
        """Retrieve all solutions."""
        return self.solutions

    def get_solution(self, solution_number: int) -> Optional[SolutionWithHistory]:
        """Retrieve a solution by its trial number."""
        return next((sol for sol in self.solutions if sol.trial_number == solution_number), None)

    def get_user_saved_solutions(self) -> list[SolutionWithHistory]:
        """Retrieve all user saved solutions."""
        return [sol for sol in self.solutions if sol.feedback and sol.feedback.save_last_solution]

    def get_last_solution(self) -> Optional[SolutionWithHistory]:
        """Retrieve the last added solution."""
        if self.solutions:
            return self.solutions[-1]
        return None

    def reset(self) -> None:
        """Clear all solutions from the pool."""
        self.solutions.clear()


_SOLUTION_POOL: SolutionPool = SolutionPool()


def get_last_solution() -> SolutionWithHistory | None:
    """Get the last added solution."""
    return _SOLUTION_POOL.get_last_solution()


def is_duplicate_solution(solution: Solution) -> bool:
    """Check if the solution is a duplicate of any existing solution in the pool."""
    return _SOLUTION_POOL.is_duplicate_solution(solution)


def add_solution(solution: SolutionWithHistory) -> None:
    """Add a new solution to the pool."""
    _SOLUTION_POOL.add_solution(solution)


def add_feedback(feedback: UserFeedback) -> None:
    """Add feedback to the last solution."""
    _SOLUTION_POOL.add_feedback(feedback)


def get_solutions() -> list[SolutionWithHistory]:
    """Get all solutions."""
    return _SOLUTION_POOL.get_solutions()


def get_all_nonempty_solutions() -> list[SolutionWithHistory]:
    """Get all solutions that have a non-empty solution field."""
    return [s for s in _SOLUTION_POOL.get_solutions() if s.solution is not None]


def get_solution(solution_number: int) -> Optional[SolutionWithHistory]:
    """Get a solution by its trial number."""
    return _SOLUTION_POOL.get_solution(solution_number)


def get_user_saved_solutions() -> list[SolutionWithHistory]:
    """Get all user saved solutions."""
    return _SOLUTION_POOL.get_user_saved_solutions()


def get_duplicate_solutions(solution: Solution) -> list[SolutionWithHistory]:
    """Get all duplicate solutions for a given solution."""
    return _SOLUTION_POOL.get_duplicate_solutions(solution)


def reset_solution_pool() -> None:
    """Clear the module-global solution pool."""
    _SOLUTION_POOL.reset()
