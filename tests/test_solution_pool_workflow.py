"""Tests for SolutionPool lifecycle workflows."""

import pytest

from paretochat.domain.problem import Solution
from paretochat.domain.scoring import ObjectiveScalarization
from paretochat.domain.solution_pool import SolutionPool, SolutionWithHistory
from paretochat.domain.user import UserFeedback, UserFeedbackSentiment


def _make_entry(profit: float, cost: float, label: str = "") -> SolutionWithHistory:
    return SolutionWithHistory(
        user_input=label or f"find profit={profit}",
        objective_scalarization=ObjectiveScalarization(),
        solution=Solution(objective_values={"profit": profit, "cost": cost}),
    )


def test_full_lifecycle():
    """Add, deduplicate, attach feedback, retrieve by trial number, then reset."""
    pool = SolutionPool()

    s1 = _make_entry(100.0, 20.0, "first")
    s2 = _make_entry(80.0, 15.0, "second")
    s3 = _make_entry(60.0, 10.0, "third")

    pool.add_solution(s1)
    pool.add_solution(s2)
    pool.add_solution(s3)

    assert s1.trial_number == 1
    assert s2.trial_number == 2
    assert s3.trial_number == 3
    assert not s1.duplicate
    assert not s2.duplicate
    assert not s3.duplicate

    # Near-duplicate of s1 (within math.isclose rel_tol=1e-4)
    dup = _make_entry(100.001, 20.0002, "duplicate of first")
    pool.add_solution(dup)
    assert dup.trial_number == 4
    assert dup.duplicate

    # Attach positive feedback with save flag to the last entry
    feedback = UserFeedback(
        sentiment=UserFeedbackSentiment.POSITIVE,
        feedback_text="great solution!",
        save_last_solution=True,
    )
    pool.add_feedback(feedback)

    saved = pool.get_user_saved_solutions()
    assert len(saved) == 1
    assert saved[0].trial_number == 4

    # Retrieve by trial number
    found = pool.get_solution(2)
    assert found is not None
    assert found.trial_number == 2
    assert found.solution.objective_values["profit"] == pytest.approx(80.0)

    # Reset clears everything
    pool.reset()
    assert pool.get_solutions() == []


def test_duplicate_detection_tolerance():
    """is_duplicate_solution uses math.isclose — values within tolerance are duplicates."""
    pool = SolutionPool()
    pool.add_solution(_make_entry(100.0, 20.0))

    # rel diff = 0.001/100 = 1e-5 < rel_tol=1e-4 → within tolerance → duplicate
    assert pool.is_duplicate_solution(Solution(objective_values={"profit": 100.001, "cost": 20.0002}))

    # rel diff = 1.5/100 = 0.015 > rel_tol=1e-4 → outside tolerance → not duplicate
    assert not pool.is_duplicate_solution(Solution(objective_values={"profit": 101.5, "cost": 20.0}))


def test_no_solution_entry_is_never_a_duplicate():
    """A SolutionWithHistory with solution=None is never flagged as a duplicate."""
    pool = SolutionPool()

    empty = SolutionWithHistory(
        user_input="no solution found",
        objective_scalarization=ObjectiveScalarization(),
        solution=None,
    )
    pool.add_solution(empty)

    assert not empty.duplicate
    assert empty.trial_number == 1
