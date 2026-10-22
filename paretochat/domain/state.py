"""State representation for the ParetoChat graph."""

from typing import Optional

from langgraph.graph import MessagesState

from paretochat.domain.agents.command_processor import UserCommand
from paretochat.domain.agents.user_intent import UserIntent
from paretochat.domain.problem import Solution
from paretochat.domain.scoring import ObjectiveScalarization
from paretochat.domain.user import UserFeedback


class State(MessagesState):
    """State representation for the ParetoChat graph.

    Inherits from MessagesState which provides a 'messages' field
    that automatically accumulates messages across nodes.
    """

    last_user_input: str
    last_user_intent: UserIntent
    last_user_command: UserCommand
    last_objective_scalarization: ObjectiveScalarization
    last_solution: Solution
    last_user_feedback: Optional[UserFeedback] = None
    found_duplicate_solution: bool = False
    num_solution_attempts: int = 0
    run_approved: bool = False
