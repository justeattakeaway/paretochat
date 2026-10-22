"""Data models needed for simulating optimization problems."""

from typing import Any

from pydantic import BaseModel

from paretochat.domain.problem import DomainInfo


class SimulationProblemInstance(BaseModel):
    """A class representing a specific instance of a simulation problem."""

    name: str
    input_data: Any
    pareto_front: list[dict[str, float]]


class SimulationProblem(BaseModel):
    """A class representing a simulation problem, including its domain info and specific instance."""

    name: str
    domain_info_path: str  # Could be removed when ParetoChatAgent is updated to accept domain info directly
    domain_info: DomainInfo
    instances: list[SimulationProblemInstance]
