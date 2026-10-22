"""Domain agents for command processing."""

from enum import Enum

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from paretochat.domain.sanitization import UNTRUSTED_INPUT_NOTICE, sanitize_user_input
from paretochat.infrastructure.llm.gemini_model import create_llm_model


class UserCommandKind(str, Enum):
    """The kind of user command."""

    FIND_NEW_SOLUTION = "FIND_NEW_SOLUTION"
    SUMMARIZE_SESSION = "SUMMARIZE_SESSION"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_str(cls):
        return ", ".join(UserCommandKind.list())


_system_message = """
You are a command processing assistant that helps to understand user commands
during the interactive session. Given the user's input, extract the main
command and the classification of the command into one of the following kinds:
{user_command_kinds}
Then, return a JSON dictionary with the fields "command" and "kind".

If the user wants to find a new solution based on their preferences,
set the "kind" to FIND_NEW_SOLUTION. The "command" field should contain
the extracted command related to finding a new solution.

If the user wants to summarize the session or see the solutions they have saved,
set the "kind" to SUMMARIZE_SESSION. The user input could be direct command or
a question related to the solutions found so far or saved by them.
"""

_human_message = """
<user_input>
{user_input}
</user_input>
"""

_template = ChatPromptTemplate(
    [
        ("system", (_system_message + UNTRUSTED_INPUT_NOTICE).format(user_command_kinds=UserCommandKind.list_str())),
        ("human", _human_message),
    ]
)


class UserCommand(BaseModel):
    """A class representing user command."""

    command: str = Field(
        description="The extracted user command from the input.",
    )
    kind: UserCommandKind = Field(
        description="The kind of the user command, e.g., FIND_NEW_SOLUTION, SUMMARIZE_SESSION.",
    )


class CommandProcessorAgent:
    """Agent to process user command."""

    def __init__(self) -> None:
        self.template = _template

        self.agent = create_agent(
            name="CommandProcessorAgent",
            model=create_llm_model(),
            tools=[],
            response_format=UserCommand,  # Auto-selects ProviderStrategy if available, otherwise ToolStrategy
        )

    def get_prompt(self, user_input: str) -> ChatPromptTemplate:
        """Get the prompt template with user input."""
        return self.template.invoke({"user_input": user_input})

    def run(self, user_input: str) -> UserCommand:
        """Run the agent with the given user input."""
        user_input = sanitize_user_input(user_input)
        prompt = self.get_prompt(user_input)
        # The end-user input in the prompt has been sanitized via sanitize_user_input above.
        result = self.agent.invoke(
            {
                "messages": prompt.messages,
            }
        )
        for m in result["messages"]:
            if m.type != "system":
                m.pretty_print()
        return result["structured_response"]
