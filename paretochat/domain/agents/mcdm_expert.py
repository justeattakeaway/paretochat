"""Agents for processing new solution commands."""

import logging

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import yaml

from paretochat.domain.agents.command_processor import UserCommand
from paretochat.domain.problem import DomainInfo, Solution
from paretochat.domain.sanitization import UNTRUSTED_INPUT_NOTICE, sanitize_user_input
from paretochat.domain.scoring import ObjectiveScalarization, ScalarizationKind
from paretochat.domain.solution_pool import (
    SolutionWithHistory,
)
from paretochat.domain.solution_pool import (
    get_all_nonempty_solutions as _get_all_nonempty_solutions,
)
from paretochat.domain.solution_pool import (
    get_duplicate_solutions as _get_duplicate_solutions,
)
from paretochat.domain.solution_pool import (
    get_last_solution as _get_last_solution,
)
from paretochat.domain.solution_pool import (
    get_solution as _get_solution,
)
from paretochat.infrastructure.llm.gemini_model import create_llm_model

_system_message = """
You are an expert on Multi Criteria Decision Making. You know what a nondominated point is.
You know what an efficient solution is. You know the different scalarization methods to solve multi-criteria
optimization problems, such as weighted sum, lexicographic and reference point-based methods.

You are given access to the following tools to get information about the problem and the previously generated solutions:
- get_domain_info: This tool returns the problem description, including the objectives and their types (minimize/maximize).
- get_all_nonempty_solutions: This tool returns all solutions previously generated that have a non-empty solution field.
- get_solution: This tool returns a solution by its trial number.
- get_last_solution: This tool returns the last added solution.
- get_duplicate_solutions: This tool returns all duplicate solutions for a given solution. The input should be a Solution
object, not a JSON or dict.
- serialize_solution: This tool serializes a Solution object into its JSON representation.
- deserialize_solution: This tool deserializes a Solution object from its JSON representation.

Your task is to analyze the user command and the given additional context, and return the implied objective scalarization.

During your analysis, you should follow the following steps:
1. Get the problem information using the get_domain_info tool. Check the `algorithm` field in the returned domain
   info:
   - `algorithm.supported_scalarizations`: you MUST only use scalarization kinds from this list.
   - `algorithm.supports_obj_bounds`: if false, you MUST leave the `bounds` field empty (no objective bounds).
2. Analyze the user request and understand which tools you need to use.
3. Extract the implied scalarization by the user command, choosing only from the supported scalarization kinds.
4. Return the scalarization in the form of ObjectiveScalarization.

If the extracted scalarization is weighted sum, you should extract a strictly positive weight for each objective in order
not to generate dominated solutions.

If the user is looking for a "similar" solution to a previous solution or something different "around" the previous
solution:
- If `algorithm.supports_obj_bounds` is true: put some reasonable bounds on the objective values to ensure that
  the new solution is "similar" to or "around" the previous solution. At the same time, change the weights
  significantly to lead to a different solution. The bounds should be around the previous solution and should be
  neither too large nor too small.
- If `algorithm.supports_obj_bounds` is false: retrieve the scalarization that led to the reference solution and
  use it as a starting point. Then adjust the weights moderately — enough to steer toward a neighbouring region of
  the Pareto front without moving too far away from the reference solution's objective values.

If the user is referencing two previous solutions and wants a new solution "between" them:
- If `algorithm.supports_obj_bounds` is true: set the bounds based on the two previous solutions so that the
  bounds define the feasible space between them. If the user also specifies some objectives for identifying the
  "in between" region, put the bounds only for those objectives. The bounds should exclude the two previous
  solutions themselves.
- If `algorithm.supports_obj_bounds` is false: retrieve the scalarizations that led to each of the two reference
  solutions. Interpolate the weights of those two scalarizations (e.g. take a weighted average) to produce weights
  that are likely to yield a solution between the two. If only specific objectives define the "in between" region,
  adjust only the weights of those objectives accordingly.

If the input tells you that the last solution was a duplicate, you should examine all the duplicate solutions of the
last solution and review the objective scalarizations that led to these duplicates. Based on this information,
you should try to extract a different scalarization that is more likely to lead to a new solution.

During your analysis, follow these guidelines:
- Always use the tools to get relevant information about the problem and previously generated solutions.
- Base your analysis on the information provided by the tools.
- If the last solution was a duplicate, analyze the duplicate solutions and their scalarizations to inform your
extraction of a new scalarization.
- If the user command is ambiguous or does not clearly indicate a scalarization method, make a reasonable assumption
based on common practices in multi-criteria decision making or the scalarizations used in the previous solutions.
- Ensure that the extracted scalarization is valid and consistent with the objectives in the problem description.
- Never make up information that is not provided by the user command or the tools.
"""

_ai_message = """
User command: <user_input>{user_command}</user_input>
Found duplicate solution: {found_duplicate_solution}
Number of solution attempts so far: {num_solution_attempts}
"""

_template = ChatPromptTemplate(
    [
        ("system", _system_message + UNTRUSTED_INPUT_NOTICE),
        ("ai", _ai_message),
    ]
)

_logger = logging.getLogger(__name__)


def make_get_domain_info_tool(domain_info_path: str):
    """Create a tool that loads domain information from the given YAML file path."""

    @tool
    def get_domain_info() -> str:
        """Get the domain information."""
        with open(domain_info_path, "r") as f:
            config = yaml.safe_load(f)
        return DomainInfo.model_validate(config).model_dump_json()

    return get_domain_info


@tool
def get_all_nonempty_solutions() -> list[SolutionWithHistory]:
    """Get all solutions that have a non-empty solution field."""
    return _get_all_nonempty_solutions()


@tool
def get_solution(solution_number: int) -> SolutionWithHistory | None:
    """Get a solution by its trial number."""
    return _get_solution(solution_number)


@tool
def get_last_solution() -> SolutionWithHistory | None:
    """Get the last added solution."""
    return _get_last_solution()


@tool
def get_duplicate_solutions(solution: Solution) -> list[SolutionWithHistory]:
    """Get all duplicate solutions for a given solution."""
    return _get_duplicate_solutions(solution)


@tool
def deserialize_solution(solution_json: str) -> SolutionWithHistory:
    """Deserialize a SolutionWithHistory object from its JSON representation."""
    return SolutionWithHistory.model_validate_json(solution_json)


@tool
def serialize_solution(solution: SolutionWithHistory) -> str:
    """Serialize a SolutionWithHistory object to its JSON representation."""
    return solution.model_dump_json()


class McdmExpertAgent:
    """Agent to process new solution request to extract the implied scalarization."""

    def __init__(self, domain_info_path: str) -> None:
        self.template = _template

        self.agent = create_agent(
            name="McdmExpertAgent",
            model=create_llm_model(),
            tools=[
                make_get_domain_info_tool(domain_info_path),
                get_all_nonempty_solutions,
                get_solution,
                get_last_solution,
                get_duplicate_solutions,
                deserialize_solution,
                serialize_solution,
            ],
            # langchain picks ToolStrategy on Gemini < 3, which can't combine tools with native structured
            # output. Gemini 3-series models automatically get ProviderStrategy.
            response_format=ObjectiveScalarization,
        )

    def get_prompt(
        self,
        user_command: UserCommand,
        found_duplicate_solution: bool = False,
        num_solution_attempts: int = 0,
    ) -> ChatPromptTemplate:
        """Get the prompt template with user input."""
        return self.template.invoke(
            {
                "user_command": user_command.model_dump_json(),
                "found_duplicate_solution": found_duplicate_solution,
                "num_solution_attempts": num_solution_attempts,
            }
        )

    def run(
        self,
        user_command: UserCommand,
        found_duplicate_solution: bool = False,
        num_solution_attempts: int = 0,
    ) -> ObjectiveScalarization:
        """Run the agent with the given user input."""
        user_command = user_command.model_copy(update={"command": sanitize_user_input(user_command.command)})
        prompt = self.get_prompt(
            user_command, found_duplicate_solution=found_duplicate_solution, num_solution_attempts=num_solution_attempts
        )
        # The end-user input in the prompt has been sanitized via sanitize_user_input above.
        result = self.agent.invoke(
            {
                "messages": prompt.messages,
            }
        )
        for m in result["messages"]:
            if m.type != "system":
                m.pretty_print()

        structured_response = result.get("structured_response")
        if structured_response is not None:
            return structured_response

        # Now retry once and if it still fails, use a fallback scalarization.
        _logger.warning(
            "McdmExpertAgent missing structured_response on first attempt. result_keys=%s user_command=%s",
            list(result.keys()),
            user_command.model_dump_json(),
        )

        retry_message = HumanMessage(
            content=(
                "Retry once. Return ONLY a valid ObjectiveScalarization as structured output. Do not return prose."
            )
        )
        # The end-user input in the prompt has been sanitized via sanitize_user_input above.
        retry_result = self.agent.invoke(
            {
                "messages": [*prompt.messages, retry_message],
            }
        )
        for m in retry_result["messages"]:
            if m.type != "system":
                m.pretty_print()

        retry_structured_response = retry_result.get("structured_response")
        if retry_structured_response is not None:
            _logger.info("McdmExpertAgent recovered structured_response on retry.")
            return retry_structured_response

        _logger.error(
            "McdmExpertAgent missing structured_response after retry; using fallback scalarization. result_keys=%s user_command=%s",
            list(retry_result.keys()),
            user_command.model_dump_json(),
        )

        # Fallback is intentionally simple and always valid.
        return ObjectiveScalarization(kind=ScalarizationKind.WEIGHTED_SUM, parameters=[], bounds=[])
