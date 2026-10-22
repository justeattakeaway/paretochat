import json
from typing import Optional, Type

from pydantic import BaseModel, Field

from paretochat.simulation.common.timed_tool import TimedTool
from paretochat.simulation.desdeo.nimbus.nimbus_simulator import (
    NimbusInput,
    NimbusMethodSimulator,
)


class DesdeoNimbusInput(BaseModel):
    classifications: Optional[dict[str, tuple[str, Optional[float]]]] = Field(
        default=None,
        description=(
            "Objective classifications as dict mapping objective symbol to a tuple of (class_type, level_or_none). "
            "Valid classes: '<' (improve), '<=' (improve until level), '=' (keep), '>=' (worsen until level), '0' (free). "
            "For '<', '=', and '0', level_or_none should be null. "
            "For '<=' and '>=', level_or_none should be a numeric value. Omit on the first call."
        ),
    )
    current_objectives: Optional[dict[str, float]] = Field(
        default=None,
        description="Current objective values that classifications are based on. Omit on the first call.",
    )


class DesdeoNimbusTool(TimedTool):
    """Tool to interact with the DESDEO NIMBUS simulator."""

    name: str = "desdeo_nimbus"
    description: str = (
        "Use this tool to generate new candidate solutions with DESDEO Synchronous NIMBUS method. "
        "NIMBUS classifies each objective (improve, improve-until, keep, worsen-until, or free). "
        "This generates 4 diverse solutions using different scalarization functions. "
        "On the first call, omit inputs to get a starting point. After that, provide classifications and current_objectives."
    )
    args_schema: Type[BaseModel] = DesdeoNimbusInput

    simulator: NimbusMethodSimulator

    class Config:
        arbitrary_types_allowed = True

    def _run_implementation(
        self,
        classifications: Optional[dict[str, tuple[str, Optional[float]]]] = None,
        current_objectives: Optional[dict[str, float]] = None,
    ) -> str:
        try:
            response = self.simulator.find_new_solutions(
                NimbusInput(
                    classifications=classifications,
                    current_objectives=current_objectives,
                )
            )
            payload = {
                "generated_solutions": [
                    {"objective_values": solution.objective_values} for solution in response.generated_solutions
                ]
            }
            return json.dumps(payload)
        except Exception as e:
            return f"Error communicating with DESDEO NIMBUS simulator: {e}"


def create_desdeo_nimbus_tool(simulator: NimbusMethodSimulator) -> DesdeoNimbusTool:
    """Factory function to create a DesdeoNimbusTool with the provided simulator."""
    return DesdeoNimbusTool(simulator=simulator)
