"""Implementation of the default state graph for ParetoChat."""

import functools
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from paretochat.domain.agents.command_processor import CommandProcessorAgent, UserCommand, UserCommandKind
from paretochat.domain.agents.feedback_processor import FeedbackProcessorAgent
from paretochat.domain.agents.mcdm_expert import McdmExpertAgent
from paretochat.domain.agents.session_summarizer import SessionSummarizerAgent
from paretochat.domain.agents.solution_presenter import SolutionPresenterAgent
from paretochat.domain.agents.user_intent import UserIntent, UserIntentAgent, UserIntentKind
from paretochat.domain.algorithm import Algorithm
from paretochat.domain.problem import ScalarizationKind, Solution
from paretochat.domain.scoring import (
    ObjectiveBound,
    ObjectiveParameter,
    ObjectiveParameterKind,
    ObjectiveScalarization,
)
from paretochat.domain.solution_pool import (
    SolutionWithHistory,
    add_feedback,
    add_solution,
    is_duplicate_solution,
)
from paretochat.domain.state import State
from paretochat.domain.user import UserFeedback, UserFeedbackSentiment

_logger = logging.getLogger(__name__)

MAX_SOLUTION_ATTEMPTS = 3

# Hard upper bound on graph steps per invocation, as a backstop to MAX_SOLUTION_ATTEMPTS.
# Longest path: intent, feedback, command, (expert, approval, algorithm) per attempt, presenter.
# Doubled to leave headroom for future nodes.
RECURSION_LIMIT = 2 * (4 + 3 * MAX_SOLUTION_ATTEMPTS)

# Retry policy for nodes that call an LLM and have no side effects before the call returns.
# The default retry_on skips programming errors (e.g. ValueError, TypeError) and HTTP 4xx errors.
LLM_NODE_RETRY_POLICY = RetryPolicy(max_attempts=3)

# Types stored in the graph state; the checkpointer may only deserialize these.
CHECKPOINT_TYPES = [
    UserIntent,
    UserIntentKind,
    UserCommand,
    UserCommandKind,
    ObjectiveScalarization,
    ObjectiveParameter,
    ObjectiveParameterKind,
    ObjectiveBound,
    ScalarizationKind,
    Solution,
    UserFeedback,
    UserFeedbackSentiment,
]


def log_node_errors(name: str, node):
    """Wrap a node so that failures are logged (without payloads) and re-raised."""

    @functools.wraps(node)
    def wrapper(state: State):
        try:
            return node(state)
        except GraphBubbleUp:
            # Control flow (e.g. interrupt for human approval), not an error.
            raise
        except Exception as e:
            _logger.exception("Node %s failed with %s", name, type(e).__name__)
            raise

    return wrapper


def user_intent_agent(state: State):
    """Wrapper to run the UserIntentAgent."""
    agent = UserIntentAgent()
    # Errors are logged and transient failures retried: this node is added via _add_logged_node with a RetryPolicy.
    user_intent = agent.run(state["last_user_input"])
    return {
        "last_user_intent": user_intent,
        "messages": [HumanMessage(content=state["last_user_input"])],
    }


def command_processor_agent(state: State):
    """Wrapper to run the CommandProcessorAgent."""
    agent = CommandProcessorAgent()
    # Errors are logged and transient failures retried: this node is added via _add_logged_node with a RetryPolicy.
    user_command = agent.run(state["last_user_input"])
    return {"last_user_command": user_command}


def feedback_processor_agent(state: State):
    """Wrapper to run the FeedbackProcessorAgent."""
    agent = FeedbackProcessorAgent()
    # Errors are logged and transient failures retried: this node is added via _add_logged_node with a RetryPolicy.
    user_feedback = agent.run(state["last_user_input"])
    add_feedback(user_feedback)
    return {"last_user_feedback": user_feedback}


def check_intent_kind(state: State) -> str:
    """Check the kind of the last user intent and determine the next node.

    Priority order when multiple intents are present:
    1. FEEDBACK is processed first
    2. COMMAND is processed after FEEDBACK (if both present)
    3. QUESTION ends the flow
    """
    user_intent = state["last_user_intent"]

    # If feedback is present, always process it first
    if user_intent.has_feedback():
        return UserIntentKind.FEEDBACK

    # Then check for command
    if user_intent.has_command():
        return UserIntentKind.COMMAND

    # Default to question (ends the flow)
    return UserIntentKind.QUESTION


def check_after_feedback(state: State) -> str:
    """Check if command should be processed after feedback.

    If the original intent included both FEEDBACK and COMMAND,
    continue to CommandProcessorAgent after processing feedback.
    """
    user_intent = state["last_user_intent"]

    if user_intent.has_command():
        return "continue_to_command"

    return "end"


def check_command_kind(state: State) -> str:
    """Check the kind of the last user command."""
    return state["last_user_command"].kind


def make_mcdm_expert_agent(domain_info_path: str):
    """Create a wrapper function to run the McdmExpertAgent."""

    def mcdm_expert_agent(state: State):
        """Wrapper to run the McdmExpertAgent."""
        agent = McdmExpertAgent(domain_info_path)
        # Errors are logged and transient failures retried: this node is added via _add_logged_node with a RetryPolicy.
        objective_scalarization = agent.run(
            state["last_user_command"],
            state.get("found_duplicate_solution", False),
            state.get("num_solution_attempts", 0),
        )
        return {"last_objective_scalarization": objective_scalarization}

    return mcdm_expert_agent


def approval_gate(state: State):
    """Pause the graph and ask the caller to approve running the algorithm.

    The LLM-produced scalarization drives the solver, so a human (or an explicit
    policy supplied by the caller) must approve it before the algorithm runs.
    Retries within the same request were already approved, so they skip the gate.
    """
    if state.get("num_solution_attempts", 0) > 0:
        return {"run_approved": True}

    approved = interrupt(
        {
            "action": "run_algorithm",
            "user_input": state["last_user_input"],
            "objective_scalarization": state["last_objective_scalarization"],
        }
    )
    update: dict = {"run_approved": bool(approved)}
    if not approved:
        update["messages"] = [AIMessage(content="Okay, I did not run the algorithm.")]
    return update


def check_run_approved(state: State) -> str:
    """Route to the algorithm only if the run was approved."""
    return "run" if state["run_approved"] else "cancel"


def make_run_algorithm(algorithm: Algorithm):
    """Create a wrapper function to run the given algorithm."""

    def run_algorithm(state: State):
        """Run the algorithm with the last objective scalarization."""

        num_solution_attempts = state.get("num_solution_attempts", 0)

        print(f"Solution attempt: {num_solution_attempts + 1}")

        objective_scalarization = state["last_objective_scalarization"]
        # The RunAlgorithm node is registered through _add_logged_node, which wraps run_algorithm in log_node_errors.
        solution = algorithm.run(objective_scalarization)

        # Careful: check for duplicate before adding the solution
        is_duplicate = is_duplicate_solution(solution) if solution is not None else False
        if is_duplicate:
            print("Found a duplicate solution.")

        add_solution(
            SolutionWithHistory(
                user_input=state["last_user_input"],
                objective_scalarization=objective_scalarization,
                solution=solution,
            )
        )

        return {
            "last_solution": solution,
            "found_duplicate_solution": is_duplicate,
            "num_solution_attempts": num_solution_attempts + 1,
        }

    return run_algorithm


def should_retry_for_another_solution(state: State) -> bool:
    """Check if the last solution is a duplicate."""
    return state["found_duplicate_solution"] and state["num_solution_attempts"] < MAX_SOLUTION_ATTEMPTS


def make_solution_presenter_agent():
    """Create a wrapper function to run the SolutionPresenterAgent."""

    def solution_presenter_agent(state: State):
        """Wrapper to run the SolutionPresenterAgent."""
        agent = SolutionPresenterAgent()
        # Errors are logged and transient failures retried: this node is added via _add_logged_node with a RetryPolicy.
        presentation_text = agent.run(
            state["last_user_input"],
            state["last_objective_scalarization"],
            state["last_solution"],
            state["found_duplicate_solution"],
        )
        return {"messages": [AIMessage(content=presentation_text)]}

    return solution_presenter_agent


def session_summarizer_agent(state: State):
    """Wrapper to run the SessionSummarizerAgent."""
    agent = SessionSummarizerAgent()
    # Errors are logged and transient failures retried: this node is added via _add_logged_node with a RetryPolicy.
    summary_text = agent.run(state["last_user_input"])
    return {"messages": [AIMessage(content=summary_text)]}


def state_printer(state: State):
    """Print the state."""
    print("State:================================================")
    for message in state["messages"]:
        message.pretty_print()
    print("=======================================================")
    return state


class ParetoChatGraph(StateGraph[State]):
    """A class representing the ParetoChat graph."""

    def __init__(
        self,
        domain_info_path: str,
        algorithm: Algorithm,
    ) -> None:
        """Initialize the ParetoChat graph."""
        super().__init__(state_schema=State)
        self.domain_info_path = domain_info_path
        self.algorithm = algorithm

    def _add_logged_node(self, name: str, node, retry_policy: RetryPolicy | None = None) -> None:
        """Add a node whose failures are logged, optionally retrying transient errors."""
        self.add_node(name, log_node_errors(name, node), retry_policy=retry_policy)

    def build(self):
        """Build the ParetoChat graph."""
        # nodes
        self._add_logged_node("UserIntentAgent", user_intent_agent, LLM_NODE_RETRY_POLICY)
        self._add_logged_node("CommandProcessorAgent", command_processor_agent, LLM_NODE_RETRY_POLICY)
        self._add_logged_node("FeedbackProcessorAgent", feedback_processor_agent, LLM_NODE_RETRY_POLICY)
        self._add_logged_node("McdmExpertAgent", make_mcdm_expert_agent(self.domain_info_path), LLM_NODE_RETRY_POLICY)
        self._add_logged_node("ApprovalGate", approval_gate)
        # No retry: the algorithm run records the solution in the pool.
        self._add_logged_node("RunAlgorithm", make_run_algorithm(self.algorithm))
        self._add_logged_node("SolutionPresenterAgent", make_solution_presenter_agent(), LLM_NODE_RETRY_POLICY)
        self._add_logged_node("SessionSummarizerAgent", session_summarizer_agent, LLM_NODE_RETRY_POLICY)
        # self.add_node("StatePrinter", state_printer)

        # edges
        # Every path ends in END. The only cycle (RunAlgorithm -> McdmExpertAgent -> ApprovalGate -> RunAlgorithm) is
        # bounded by MAX_SOLUTION_ATTEMPTS in should_retry_for_another_solution, with RECURSION_LIMIT as a backstop.
        self.add_edge(START, "UserIntentAgent")
        self.add_conditional_edges(
            "UserIntentAgent",
            check_intent_kind,
            {
                UserIntentKind.COMMAND: "CommandProcessorAgent",
                UserIntentKind.FEEDBACK: "FeedbackProcessorAgent",
                # TODO: implement question handling.
                UserIntentKind.QUESTION: END,
            },
        )
        self.add_conditional_edges(
            "CommandProcessorAgent",
            check_command_kind,
            {
                UserCommandKind.FIND_NEW_SOLUTION: "McdmExpertAgent",
                UserCommandKind.SUMMARIZE_SESSION: "SessionSummarizerAgent",
            },
        )
        # After feedback, check if command should also be processed
        self.add_conditional_edges(
            "FeedbackProcessorAgent",
            check_after_feedback,
            {
                "continue_to_command": "CommandProcessorAgent",
                "end": END,
            },
        )
        self.add_edge("McdmExpertAgent", "ApprovalGate")
        self.add_conditional_edges(
            "ApprovalGate",
            check_run_approved,
            {"run": "RunAlgorithm", "cancel": END},
        )
        self.add_conditional_edges(
            "RunAlgorithm",
            should_retry_for_another_solution,
            {
                True: "McdmExpertAgent",  # If duplicate, run MCDM expert again
                False: "SolutionPresenterAgent",
            },
        )
        # self.add_edge("SolutionPresenterAgent", "StatePrinter")
        # self.add_edge("StatePrinter", END)
        self.add_edge("SolutionPresenterAgent", END)
        self.add_edge("SessionSummarizerAgent", END)

        # The checkpointer is required for interrupt(); state is kept per thread_id.
        # Human approval is implemented by the ApprovalGate node, which calls interrupt() before RunAlgorithm.
        try:
            compiled_graph = self.compile(
                checkpointer=InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)),
            )
        except Exception:
            _logger.exception("Failed to compile the ParetoChat graph")
            raise
        _logger.info("ParetoChat graph compiled")
        return compiled_graph
