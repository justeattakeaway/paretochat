"""Tests for validation of the approval decision."""

import pytest

from paretochat.domain.agents.agent import validate_approval


def test_keeps_real_bool_decisions():
    assert validate_approval(True) is True
    assert validate_approval(False) is False


@pytest.mark.parametrize("decision", ["no", "False", "", "yes", 1, 0, None, {"approved": False}])
def test_non_bool_decisions_are_denied(decision):
    assert validate_approval(decision) is False
