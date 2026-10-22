"""Implementation of an AI Decision Maker agent using LangChain."""

from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Type

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from paretochat.domain.problem import ProblemInfo, Solution
from paretochat.infrastructure.llm.gemini_model import create_llm_model


class EvaluateSolutionInput(BaseModel):
    solution: Solution = Field(description="The solution to evaluate, containing objective values.")


class EvaluateSolutionOutput(BaseModel):
    score: float = Field(
        description="The scalar score of the solution based on the preference function. Higher is better."
    )


class EvaluateSolutionTool(BaseTool):
    """Tool to evaluate a solution using the preference function."""

    name: str = "evaluate_solution"
    description: str = "Use this tool to evaluate a solution's objectives. Returns a scalar score."
    args_schema: Type[BaseModel] = EvaluateSolutionInput

    preference_func: Callable[[Dict[str, float]], float]

    num_evaluations: int = 0

    class Config:
        arbitrary_types_allowed = True

    def _run(self, solution: Solution) -> EvaluateSolutionOutput:
        self.num_evaluations += 1
        try:
            score = self.preference_func(solution.objective_values)
            return EvaluateSolutionOutput(score=score)
        except Exception as e:
            print(f"Error evaluating solution: {e}")
            return EvaluateSolutionOutput(score=float("-inf"))


class Decision(BaseModel):
    solution: Solution = Field(description="The chosen solution.")
    reasoning: str = Field(description="Why this solution was chosen.")


class UpdateDecisionTool(BaseTool):
    """Tool to update the final decision on the most preferred solution."""

    name: str = "update_decision"
    description: str = "Use this tool to update the final decision on the most preferred solution."
    args_schema: Type[BaseModel] = Decision

    # The last decision stores the most preferred solution and reasoning. Each call to this tool updates it.
    decisions: list[Decision] = []

    class Config:
        arbitrary_types_allowed = True

    def _run(self, solution: Solution, reasoning: str) -> str:
        self.decisions.append(Decision(solution=solution, reasoning=reasoning))
        return f"Most preferred solution is updated with the new solution {self.decisions[-1]}."


class SimulationResult(BaseModel):
    num_eval_sols: int = 0  # of solutions evaluated
    history: List[Decision] = []
    final_decision: Optional[Decision] = None
    total_dm_time_seconds: float = 0.0
    solver_time_seconds: float = 0.0
    non_solver_time_seconds: float = 0.0
    solver_call_count: int = 0
    session_messages: List[str] = []


_system_message = """
You are a decision maker trying to find your objective values that optimizes your internal preference function.

The problem has the following objectives: {objectives_str}. Note that for maximization objectives,
higher objective values are better, while for minimization objectives, lower objective values are better.

Your goal is to find a solution that optimizes your internal preference function, which you can evaluate using the evaluate_solution tool.
The evaluate_solution tool returns a scalar score representing how much you prefer that solution.
Higher score means you prefer that solution more. You should maximize this score to find your most preferred solution.

You mimic the behaviour of a human decision maker who is trying to find their most preferred solution for a multi-objective optimization problem.
Typically, you will go through the following two main phases:
1. Learning (Exploration) phase:
- Explore the entire objective space by asking {solution_tool_name} for solutions with different preferences to learn how the objective values affect your preference function.
- Build an understanding of the trade-offs between different objectives and how they contribute to your overall preference.
- Identify the potential regions in the objective space where your most preferred solution might lie.
2. Exploitation phase:
- Focus on the promising regions of the objective space identified during the learning phase.
- Refine your queries to {solution_tool_name} to find better solutions within those regions.
- Guide the search towards most preferred solutions.

Instructions:
1. Use {solution_tool_name} to find the solutions. You can find the user manual below.
2. When you receive a solution from {solution_tool_name}, use evaluate_solution to get its score.
3. If this is the first solution or the new solution is better than previous ones, use update_decision to update the most preferred solution.
4. To explore further, refine your query to {solution_tool_name} to search for better solutions.
5. When {solution_tool_name} asks for feedback, provide it based on the scores you get from evaluate_solution to guide it towards better solutions.
6. Repeat until satisfied or you run out of ideas. You have a maximum of {max_iterations} iterations to find the best solution. Focus on finding the best solution within the iteration limit rather than trying to be perfect. If you cannot find a better solution, finalize your decision with the best one you have found so far.
7. If your exploitation stucks somewhere, implying no significant value improvements for a while, go back to exploration phase and repeat the exploration - exploitation cycle.
8. If you are not getting a new solution at an iteration, do not use the same query in the next iteration. Change something.
9. Use {solution_tool_name} only to ask for new solutions. Do not use its other features like session summarization.
10. Terminate your interaction with {solution_tool_name} gracefully when you are done.

You must follow this user guide to interact with {solution_tool_name} effectively:
{user_guide}
"""


class AIDecisionMaker:
    """An AI agent that acts as a decision maker using the provided tools."""

    def __init__(
        self,
        problem_info: ProblemInfo,
        preference_function: Callable[[Dict[str, float]], float],
    ):
        self.problem_info = problem_info
        self.preference_function = preference_function

    def run(self, solution_tool: BaseTool, max_iterations: int = 15) -> SimulationResult:
        """Run the decision maker agent loop."""

        simulation_result = SimulationResult()

        update_decision_tool = UpdateDecisionTool()
        evaluate_solution_tool = EvaluateSolutionTool(preference_func=self.preference_function)
        tools = [solution_tool, evaluate_solution_tool, update_decision_tool]

        objectives_str = ", ".join([f"{obj.name} ({obj.sense})" for obj in self.problem_info.objectives])
        user_guide = self._load_user_guide(solution_tool)
        system_message = _system_message.format(
            objectives_str=objectives_str,
            solution_tool_name=solution_tool.name,  # use the provided solution tool
            user_guide=user_guide,
            max_iterations=max_iterations,
        )
        human_message = (
            "Start finding the best solution for this problem according to my preferences. Use the available tools."
        )

        llm = create_llm_model()
        agent = create_agent(
            name="AIDecisionMaker",
            model=llm,
            tools=tools,
            system_prompt=system_message,
        )

        print("Starting AI Decision Maker...")
        run_start_time = perf_counter()
        session_messages: list[str] = []

        try:
            config = {
                "recursion_limit": max_iterations * 10
            }  # Set a recursion limit to prevent infinite loops, allowing some buffer for tool calls
            inputs = {
                "messages": [
                    HumanMessage(content=human_message),
                ]
            }

            num_printed = 0

            for state in agent.stream(inputs, config=config, stream_mode="values"):
                messages = state.get("messages", [])
                if len(messages) > num_printed:
                    for msg in messages[num_printed:]:
                        session_messages.append(self._format_message(msg))
                        msg.pretty_print()
                    num_printed = len(messages)

        except GraphRecursionError:
            print("Max iterations reached without a final decision.")

        except Exception as e:
            print(f"Error during execution: {e}")

        # populate the simulation result
        total_dm_time_seconds = perf_counter() - run_start_time
        solver_time_seconds = float(getattr(solution_tool, "total_execution_time_seconds", 0.0))
        solver_call_count = int(getattr(solution_tool, "execution_count", 0))

        simulation_result.num_eval_sols = evaluate_solution_tool.num_evaluations
        simulation_result.history = update_decision_tool.decisions
        if update_decision_tool.decisions:
            simulation_result.final_decision = update_decision_tool.decisions[-1]
        simulation_result.total_dm_time_seconds = total_dm_time_seconds
        simulation_result.solver_time_seconds = solver_time_seconds
        simulation_result.non_solver_time_seconds = max(total_dm_time_seconds - solver_time_seconds, 0.0)
        simulation_result.solver_call_count = solver_call_count
        simulation_result.session_messages = session_messages

        print(f"AI Decision Maker run completed with final decision {simulation_result.final_decision}.")
        return simulation_result

    def _format_message(self, message: BaseMessage) -> str:
        """Format a streamed message into a simple log line."""
        role = getattr(message, "type", message.__class__.__name__)
        content = self._content_to_text(getattr(message, "content", ""))

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            return f"[{role}] {content}\nTOOL_CALLS: {tool_calls}"

        return f"[{role}] {content}"

    def _content_to_text(self, content: object) -> str:
        """Convert model message content into plain text for logging."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                    else:
                        parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(parts)

        return str(content)

    def _load_user_guide(self, solution_tool: BaseTool) -> str:
        """Load user guide for the provided solution tool."""

        if solution_tool.name == "paretochat":
            guide_path = Path(__file__).parent / "paretochat/user_guide.md"
        elif solution_tool.name == "desdeo_rpm":
            guide_path = Path(__file__).parent / "desdeo/rpm/user_guide.md"
        elif solution_tool.name == "desdeo_nimbus":
            guide_path = Path(__file__).parent / "desdeo/nimbus/user_guide.md"
        else:
            raise ValueError(f"Unknown solution tool: {solution_tool.name}")

        return guide_path.read_text()
