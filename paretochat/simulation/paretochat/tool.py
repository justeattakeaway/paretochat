from typing import Type

from pydantic import BaseModel, Field

from paretochat.domain.agents.agent import ParetoChatAgent
from paretochat.simulation.common.timed_tool import TimedTool


class ParetoChatInput(BaseModel):
    query: str = Field(description="The natural language query to send to ParetoChat agent.")


class ParetoChatTool(TimedTool):
    """Tool to interact with ParetoChatAgent."""

    name: str = "paretochat"
    description: str = "Use this tool to ask ParetoChat to find a solution. Input should be a natural language query describing optimization preferences."
    args_schema: Type[BaseModel] = ParetoChatInput

    agent: ParetoChatAgent

    class Config:
        arbitrary_types_allowed = True

    def _run_implementation(self, query: str) -> str:
        try:
            response = self.agent.run(query)
            return response
        except Exception as e:
            return f"Error communicating with ParetoChat: {e}"


def create_paretochat_tool(agent: ParetoChatAgent) -> ParetoChatTool:
    """Factory function to create an ParetoChatTool with the provided agent."""
    return ParetoChatTool(agent=agent)
