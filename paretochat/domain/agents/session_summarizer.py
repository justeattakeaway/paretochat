"""Agent for summarizing the session progress to the user."""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

from paretochat.domain.sanitization import UNTRUSTED_INPUT_NOTICE, sanitize_user_input
from paretochat.domain.solution_pool import (
    get_last_solution,
    get_solutions,
    get_user_saved_solutions,
)
from paretochat.infrastructure.llm.gemini_model import create_llm_model

_system_message = """
You are a session summarizer that helps users understand their progress in exploring multi-criteria optimization solutions.

You have access to the following tools to retrieve information about the session:
- get_all_solutions: Returns all solutions generated during this session, including user inputs, objective scalarizations, and feedback.
- get_saved_solutions: Returns only the solutions that the user has explicitly saved.
- get_current_solution: Returns the most recent solution generated.

Your task is to provide a clear and concise summary of the session progress to the user.

During your summary, follow these guidelines:
1. First, use the tools to gather information about the session.
2. Provide an overview of how many solutions have been explored and how many have been saved.
3. Highlight the key decisions and feedback the user has provided.
4. Summarize the trade-offs explored across different objectives.
5. If there are saved solutions, compare them briefly to show the range of options.
6. Keep the summary concise but informative.
7. Use clear formatting to make the summary easy to read.
8. End with a brief note on potential next steps the user might consider.
9. Do not generate any additional solution numbersin the summary, use trial number instead.
"""

_ai_message = """
Please provide a summary of the session progress so far.
Additional context from user (if any): <user_input>{user_request}</user_input>
"""

_template = ChatPromptTemplate(
    [
        ("system", _system_message + UNTRUSTED_INPUT_NOTICE),
        ("ai", _ai_message),
    ]
)


@tool
def get_all_solutions() -> str:
    """Get all solutions generated during this session with their history."""
    solutions = get_solutions()
    if len(solutions) == 0:
        return "No solutions have been generated in this session yet."

    solutions_str = "\n\n".join(
        [
            f"Trial {s.trial_number}:\n"
            f"  User Input: {s.user_input}\n"
            f"  Objective Scalarization: {s.objective_scalarization.model_dump_json() if s.objective_scalarization else 'N/A'}\n"
            f"  Solution: {s.solution.model_dump_json() if s.solution else 'No solution found'}\n"
            f"  Feedback: {s.feedback.model_dump_json() if s.feedback else 'No feedback yet'}"
            for s in solutions
        ]
    )
    return f"All Solutions ({len(solutions)} total):\n\n{solutions_str}"


@tool
def get_saved_solutions() -> str:
    """Get all solutions that the user has saved."""
    solutions = get_user_saved_solutions()
    if len(solutions) == 0:
        return "No solutions have been saved by the user yet."

    solutions_str = "\n\n".join(
        [
            f"Saved Solution (Trial {s.trial_number}):\n"
            f"  User Input: {s.user_input}\n"
            f"  Objective Scalarization: {s.objective_scalarization.model_dump_json() if s.objective_scalarization else 'N/A'}\n"
            f"  Solution: {s.solution.model_dump_json() if s.solution else 'No solution found'}\n"
            f"  Feedback: {s.feedback.model_dump_json() if s.feedback else 'No feedback'}"
            for s in solutions
        ]
    )
    return f"Saved Solutions ({len(solutions)} total):\n\n{solutions_str}"


@tool
def get_current_solution() -> str:
    """Get the most recent solution generated in this session."""
    solution = get_last_solution()
    if solution is None:
        return "No solutions have been generated yet."

    return (
        f"Current Solution (Trial {solution.trial_number}):\n"
        f"  User Input: {solution.user_input}\n"
        f"  Objective Scalarization: {solution.objective_scalarization.model_dump_json() if solution.objective_scalarization else 'N/A'}\n"
        f"  Solution: {solution.solution.model_dump_json() if solution.solution else 'No solution found'}\n"
        f"  Feedback: {solution.feedback.model_dump_json() if solution.feedback else 'No feedback yet'}"
    )


class SessionSummarizerAgent:
    """Agent to summarize the session progress to the user."""

    def __init__(self) -> None:
        self.template = _template

        self.agent = create_agent(
            name="SessionSummarizerAgent",
            model=create_llm_model(),
            tools=[
                get_all_solutions,
                get_saved_solutions,
                get_current_solution,
            ],
        )

    def get_prompt(self, user_request: str = "") -> ChatPromptTemplate:
        """Get the prompt template with optional user request context."""
        return self.template.invoke({"user_request": user_request})

    def run(self, user_request: str = "") -> str:
        """Run the agent to generate a session summary.

        Args:
            user_request: Optional additional context or specific aspects the user wants summarized.

        Returns:
            A formatted summary of the session progress.
        """
        user_request = sanitize_user_input(user_request)
        prompt = self.get_prompt(user_request)
        # The end-user input in the prompt has been sanitized via sanitize_user_input above.
        result = self.agent.invoke({"messages": prompt.messages})
        for m in result["messages"]:
            if m.type != "system":
                m.pretty_print()
        content = result["messages"][-1].content
        # Handle case where content is a list (e.g., from Gemini multi-part responses)
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif part:
                    text_parts.append(str(part))
            return "\n".join(text_parts)
        return content
