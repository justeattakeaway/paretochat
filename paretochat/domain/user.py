"""Domain models for a user."""

from enum import Enum

from pydantic import BaseModel, Field


class UserFeedbackSentiment(str, Enum):
    """The sentiment of user feedback."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def list_str(cls):
        return ", ".join(UserFeedbackSentiment.list())


class UserFeedback(BaseModel):
    """A class representing user feedback on a solution."""

    sentiment: UserFeedbackSentiment
    feedback_text: str
    save_last_solution: bool = Field(
        default=False,
        description="Indicates whether the user wants to save the last solution.",
    )
