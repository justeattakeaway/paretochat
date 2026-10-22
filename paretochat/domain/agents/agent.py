"""Implementation of the ParetoChat agent."""

import logging
from typing import Callable
import uuid

from langgraph.types import Command

from paretochat.domain.algorithm import Algorithm
from paretochat.domain.graph import RECURSION_LIMIT, ParetoChatGraph
from paretochat.domain.sanitization import sanitize_user_input

_logger = logging.getLogger(__name__)

# Receives the interrupt payload (action, user input, scalarization) and returns True to approve.
ApprovalHandler = Callable[[dict], bool]


def approve_all(_request: dict) -> bool:
    """Approval policy for non-interactive runs (e.g., simulations)."""
    return True


def validate_approval(decision: object) -> bool:
    """Return the approval decision, treating anything other than a real bool as a denial."""
    if isinstance(decision, bool):
        return decision
    _logger.warning("Approval handler returned %s instead of bool; treating as denied", type(decision).__name__)
    return False


class ParetoChatAgent:
    """A class representing the ParetoChat agent."""

    def __init__(
        self,
        domain_info_path: str,
        algorithm: Algorithm,
        approval_handler: ApprovalHandler = approve_all,
    ) -> None:
        """Initialize the ParetoChat agent.

        Args:
            approval_handler: Called before the algorithm is run with the proposed
                scalarization. Return False to cancel. Interactive front ends should
                pass a handler that asks the user; non-interactive runs may keep the default.
        """
        self.graph = ParetoChatGraph(domain_info_path, algorithm).build()
        self.approval_handler = approval_handler
        self._config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": RECURSION_LIMIT,
        }

    def run(self, user_input: str):
        """Run the ParetoChat agent.

        The graph checkpointer persists the state across invocations to maintain
        context (e.g., last_solution for feedback processing).
        """
        result = self.graph.invoke(
            {"last_user_input": sanitize_user_input(user_input), "num_solution_attempts": 0},
            self._config,
        )

        # Resume as long as the graph is waiting for an approval decision.
        while result.get("__interrupt__"):
            approved = validate_approval(self.approval_handler(result["__interrupt__"][0].value))
            # The approval decision has been validated via validate_approval above: only a real bool can approve.
            result = self.graph.invoke(Command(resume=approved), self._config)

        return result
