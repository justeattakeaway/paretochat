import json
from typing import Optional, Type

from pydantic import BaseModel, Field

from paretochat.simulation.common.timed_tool import TimedTool
from paretochat.simulation.desdeo.rpm.rpm_simulator import ReferencePointMethodSimulator, RPMInput


class DesdeoRpmInput(BaseModel):
    reference_point: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "Reference point values keyed by objective symbol (for example, {'f_1': 10.0, 'f_2': 20.0}). "
            "Use null to request the initial iteration around the ideal point."
        ),
    )


class DesdeoRpmTool(TimedTool):
    """Tool to interact with the DESDEO RPM simulator."""

    name: str = "desdeo_rpm"
    description: str = (
        "Use this tool to generate new candidate solutions with DESDEO Reference Point Method. "
        "Input should provide a reference_point dictionary or null for the first iteration."
    )
    args_schema: Type[BaseModel] = DesdeoRpmInput

    simulator: ReferencePointMethodSimulator

    class Config:
        arbitrary_types_allowed = True

    def _run_implementation(self, reference_point: Optional[dict[str, float]] = None) -> str:
        try:
            response = self.simulator.find_new_solutions(RPMInput(reference_point=reference_point))
            payload = {
                "generated_solutions": [
                    {"objective_values": solution.objective_values} for solution in response.generated_solutions
                ]
            }
            return json.dumps(payload)
        except Exception as e:
            return f"Error communicating with DESDEO RPM simulator: {e}"


def create_desdeo_rpm_tool(simulator: ReferencePointMethodSimulator) -> DesdeoRpmTool:
    """Factory function to create a DesdeoRpmTool with the provided simulator."""
    return DesdeoRpmTool(simulator=simulator)
