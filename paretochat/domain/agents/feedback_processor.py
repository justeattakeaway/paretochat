"""Domain agents for feedback processing."""

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

from paretochat.domain.sanitization import UNTRUSTED_INPUT_NOTICE, sanitize_user_input
from paretochat.domain.user import UserFeedback, UserFeedbackSentiment
from paretochat.infrastructure.llm.gemini_model import create_llm_model

_system_message = """
You are a feedback processing assistant that helps to understand user feedback
during the interactive session. During the session, the user is presented with different solutions
and asked to provide feedback on them and if they want to save them.

Given the user's input, extract the main sentiment and the classification of the sentiment into one
of the following kinds: {user_sentiment_kinds}. Also understand if this is a feedback on a particular
solution. If the user wants to save the solution, set the "save_last_solution" field to true. Even
if the user did not specifically ask to save the solution, set the "save_last_solution" field to
true only if the sentiment is positive. Otherwise, set the "save_last_solution" field to false.

Return a UserFeedback with fields "sentiment", "feedback_text" and "save_last_solution".

Before returning the response, write a message to the user that you have received their feedback and
they can continue the session either by asking for a new solution or by asking to summarize the session.
"""

_human_message = """
<user_input>
{user_input}
</user_input>
"""

_template = ChatPromptTemplate(
    [
        (
            "system",
            (_system_message + UNTRUSTED_INPUT_NOTICE).format(user_sentiment_kinds=UserFeedbackSentiment.list_str()),
        ),
        ("human", _human_message),
    ]
)


class FeedbackProcessorAgent:
    """Agent to process user feedback."""

    def __init__(self) -> None:
        self.template = _template

        self.agent = create_agent(
            name="FeedbackProcessorAgent",
            model=create_llm_model(),
            tools=[],
            response_format=UserFeedback,
        )

    def get_prompt(self, user_input: str) -> ChatPromptTemplate:
        """Get the prompt template with user input."""
        return self.template.invoke({"user_input": user_input})

    def run(self, user_input: str) -> UserFeedback:
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
