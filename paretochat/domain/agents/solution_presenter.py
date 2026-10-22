"""Agents for presenting solutions to the user."""

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

from paretochat.domain.problem import Solution
from paretochat.domain.sanitization import UNTRUSTED_INPUT_NOTICE, sanitize_user_input
from paretochat.domain.scoring import ObjectiveScalarization
from paretochat.infrastructure.llm.gemini_model import create_llm_model

_system_message = """
You are a solution presenter that helps to present solutions to the user.
You know the different scalarization methods to solve multi-criteria optimization problems, such as weighted sum, lexicographic and reference point-based methods.

You are given the following information:
- User input: The user input that led to the solution.
- Objective scalarization: The objective scalarization that led to the solution.
- Solution: The solution itself.

Your task is to present the solution and the objective scalarization to the user.

During your presentation, follow these guidelines:
- Restrict your presentation to the information provided in the user input, objective scalarization and solution.
- Present the solution in a way that is easy to understand for the user.
- Present the objective scalarization in a way that is easy to understand for the user. Do not skip any information about the objective scalarization, as the user may not be familiar with the concept of objective scalarization and may need all the information to understand the solution.
- If there was a solution, ask at the end of the presentation if the user has any feedback on the solution and if they want to save it or not.
- If the solution is a duplicate, inform the user that no other solution could be found and the presented solution is the same as one of the previously generated solutions.
"""

_ai_message = """
User input: <user_input>{user_input}</user_input>
Objective scalarization: {objective_scalarization}
Solution: {solution}
Found duplicate solution: {found_duplicate_solution}
"""

_template = ChatPromptTemplate(
    [
        ("system", _system_message + UNTRUSTED_INPUT_NOTICE),
        ("ai", _ai_message),
    ]
)


class SolutionPresenterAgent:
    """Agent to present solutions to the user."""

    def __init__(self) -> None:
        self.template = _template

        self.agent = create_agent(
            name="SolutionPresenterAgent",
            model=create_llm_model(),
            tools=[],
        )

    def get_prompt(
        self,
        user_input: str,
        objective_scalarization: ObjectiveScalarization,
        solution: Solution,
        found_duplicate_solution: bool = False,
    ) -> ChatPromptTemplate:
        """Get the prompt template with user input, objective scalarization and solution."""
        return self.template.invoke(
            {
                "user_input": user_input,
                "objective_scalarization": objective_scalarization,
                "solution": solution,
                "found_duplicate_solution": found_duplicate_solution,
            }
        )

    def run(
        self,
        user_input: str,
        objective_scalarization: ObjectiveScalarization,
        solution: Solution,
        found_duplicate_solution: bool = False,
    ) -> str:
        """Run the agent with the given user input, objective scalarization and solution."""
        user_input = sanitize_user_input(user_input)
        prompt = self.get_prompt(user_input, objective_scalarization, solution, found_duplicate_solution)
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
