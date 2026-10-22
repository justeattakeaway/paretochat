"""Domain agents for user intent processing."""

from enum import Enum

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from paretochat.domain.sanitization import UNTRUSTED_INPUT_NOTICE, sanitize_user_input
from paretochat.infrastructure.llm.gemini_model import create_llm_model


class UserIntentKind(str, Enum):
    """The kind of user intent."""

    QUESTION = "QUESTION"
    COMMAND = "COMMAND"
    FEEDBACK = "FEEDBACK"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_str(cls):
        return ", ".join(UserIntentKind.list())


_system_message = """
You are a communication assistant that helps to understand user intent
during the interactive session. Given the user's input, extract the main
intent and classify it into one or more of the following kinds:
{user_intent_kinds}
Then, return a UserIntent with the fields "intent" and "kinds".

IMPORTANT: A user's input may contain MULTIPLE intents. For example, the user might
provide feedback on the current solution AND request a new solution in the same message.
In such cases, include ALL applicable intent kinds in the "kinds" list.

If the user is asking a question, check if the question is about whether such a
solution exists or what is the minimum or maximum achievable value for an objective.
If it is, the intent is to find a solution and COMMAND should be included in "kinds".

Note that if the user wants to save the solution, they will provide feedback on the solution and ask to save it.
In this case, FEEDBACK should be included in "kinds". User request to save or not to save a solution is part of
the feedback and should not be treated as a separate command.

Note that the current capability supports feedback input only on the last solution presented to the user.
If the user is referring to another previously generated solution, or if you are not entirely sure that the user is referring to the last solution, do not generate any feedback as an intent.

Examples of multiple intents:
- "I like this solution, save it. Now find me a cheaper option." -> kinds: [FEEDBACK, COMMAND]
- "This looks good, please save it and show me alternatives." -> kinds: [FEEDBACK, COMMAND]
- "This is not what I was looking for, find me a cheaper option." -> kinds: [FEEDBACK, COMMAND]
"""

_human_message = """
<user_input>
{user_input}
</user_input>
"""

_template = ChatPromptTemplate(
    [
        ("system", (_system_message + UNTRUSTED_INPUT_NOTICE).format(user_intent_kinds=UserIntentKind.list_str())),
        ("human", _human_message),
    ]
)


class UserIntent(BaseModel):
    """A class representing user intent."""

    intent: str = Field(
        description="The extracted user intent from the input.",
    )
    kinds: list[UserIntentKind] = Field(
        description="The list of intent kinds detected in the user input. May contain multiple kinds if the user expresses multiple intents (e.g., [FEEDBACK, COMMAND]).",
    )

    def has_feedback(self) -> bool:
        """Check if the intent includes feedback."""
        return UserIntentKind.FEEDBACK in self.kinds

    def has_command(self) -> bool:
        """Check if the intent includes a command."""
        return UserIntentKind.COMMAND in self.kinds

    def has_question(self) -> bool:
        """Check if the intent includes a question."""
        return UserIntentKind.QUESTION in self.kinds


class UserIntentAgent:
    """Agent to process user intent."""

    def __init__(self) -> None:
        self.template = _template

        self.agent = create_agent(
            name="UserIntentAgent",
            model=create_llm_model(),
            tools=[],
            response_format=UserIntent,
        )

    def get_prompt(self, user_input: str) -> ChatPromptTemplate:
        """Get the prompt template with user input."""
        return self.template.invoke({"user_input": user_input})

    def run(self, user_input: str) -> UserIntent:
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
