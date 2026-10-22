"""This module provides factory methods to create DESDEO Problem instances from a SimulationProblem instances."""

from desdeo.problem import (
    Constraint,
    ConstraintTypeEnum,
    Objective,
    Problem,
    Variable,
    VariableTypeEnum,
)

from paretochat.domain.problem import ObjectiveSense
from paretochat.simulation.knapsack import KnapsackSimulationProblem
from paretochat.simulation.simulation_problem import SimulationProblemInstance
from paretochat.simulation.spa import SPASimulationProblem


def create_knapsack_problem(
    simulation_problem: KnapsackSimulationProblem, instance: SimulationProblemInstance
) -> Problem:
    """
    Create a DESDEO knapsack problem from a KnapsackSimulationProblem instance.

    This ensures consistency between the simulation problem's domain info (including objective
    definitions and names) and the DESDEO problem representation.

    Args:
        simulation_problem: A KnapsackSimulationProblem instance with populated domain_info.
        instance: A SimulationProblemInstance containing the specific input data for this problem instance.

    Returns:
        A DESDEO Problem instance with objectives matching the simulation problem's domain info.

    Example:
        >>> from paretochat.simulation.knapsack import KnapsackSimulationProblem
        >>> sim_problem = KnapsackSimulationProblem(base_path="path_to_instances")
        >>> instance = sim_problem.instances[0]
        >>> problem = create_knapsack_problem(sim_problem, instance)
        >>> from desdeo.tools import payoff_table_method
        >>> ideal, nadir = payoff_table_method(problem)
        >>> problem = problem.update_ideal_and_nadir(new_ideal=ideal, new_nadir=nadir)
    """
    input_data = instance.input_data
    num_items = input_data.num_items
    capacity = input_data.capacity
    items = input_data.items

    objective_info_list = simulation_problem.domain_info.problem.objectives

    # Step 1: Create decision variables (one per item, binary as integer 0-1)
    variables = []
    for i in range(num_items):
        var = Variable(
            name=f"Item {i}",
            symbol=f"x_{i}",
            variable_type=VariableTypeEnum.binary,
            lowerbound=0,
            upperbound=1,
            initial_value=0,
        )
        variables.append(var)

    # Step 2: Create objective functions from simulation problem domain info
    objectives = []
    for j, obj_info in enumerate(objective_info_list):
        # Build the objective function expression as a sum of item contributions
        # Format: x_0*value_0 + x_1*value_1 + ...
        terms = []
        for i in range(num_items):
            obj_value = items[i][1][j]
            terms.append(f"x_{i}*{obj_value}")
        func_expr = " + ".join(terms)

        # Use objective info name and sense from domain_info
        maximize = obj_info.sense == ObjectiveSense.MAXIMIZE

        obj = Objective(
            name=obj_info.name,
            symbol=obj_info.name,
            func=func_expr,
            maximize=maximize,
            is_convex=True,
            is_linear=True,
            is_twice_differentiable=True,
        )
        objectives.append(obj)

    # Step 3: Create capacity constraint (standard form: g <= 0)
    # Constraint: sum(x_i * weight_i) - capacity <= 0
    constraint_terms = []
    for i in range(num_items):
        weight = items[i][0]
        constraint_terms.append(f"x_{i}*{weight}")
    constraint_func = " + ".join(constraint_terms) + f" - {capacity}"

    capacity_constraint = Constraint(
        name="Knapsack Capacity",
        symbol="g_capacity",
        func=constraint_func,
        cons_type=ConstraintTypeEnum.LTE,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )

    # Step 4: Create and return the Problem
    problem = Problem(
        name=f"Multi-objective Knapsack Problem: {simulation_problem.name}",
        description=simulation_problem.domain_info.problem.description,
        constants=[],
        variables=variables,
        objectives=objectives,
        constraints=[capacity_constraint],
    )

    return problem


def create_spa_problem(
    simulation_problem: SPASimulationProblem,
    instance: SimulationProblemInstance,
) -> Problem:
    """Create a DESDEO set-partitioning problem from an SPA simulation instance."""
    input_data = instance.input_data
    num_rows = input_data.num_rows
    num_columns = input_data.num_columns
    column_costs = input_data.column_costs
    covered_rows_by_column = input_data.covered_rows_by_column

    objective_info_list = simulation_problem.domain_info.problem.objectives

    variables = []
    for i in range(num_columns):
        variables.append(
            Variable(
                name=f"Column {i}",
                symbol=f"x_{i}",
                variable_type=VariableTypeEnum.binary,
                lowerbound=0,
                upperbound=1,
                initial_value=0,
            )
        )

    objectives = []
    for j, obj_info in enumerate(objective_info_list):
        terms = [f"x_{i}*{column_costs[i][j]}" for i in range(num_columns)]
        func_expr = " + ".join(terms) if terms else "0"
        objectives.append(
            Objective(
                name=obj_info.name,
                symbol=obj_info.name,
                func=func_expr,
                maximize=obj_info.sense == ObjectiveSense.MAXIMIZE,
                is_convex=True,
                is_linear=True,
                is_twice_differentiable=True,
            )
        )

    constraints = []
    for row_idx in range(num_rows):
        covering_columns = [
            col_idx for col_idx, covered_rows in enumerate(covered_rows_by_column) if row_idx in covered_rows
        ]
        if not covering_columns:
            raise ValueError(f"Row {row_idx} has no covering column; instance is infeasible.")

        lhs_terms = [f"x_{col_idx}" for col_idx in covering_columns]
        constraint_func = f"{' + '.join(lhs_terms)} - 1"
        constraints.append(
            Constraint(
                name=f"Row Coverage {row_idx}",
                symbol=f"g_row_{row_idx}",
                func=constraint_func,
                cons_type=ConstraintTypeEnum.EQ,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        )

    return Problem(
        name=f"Multi-objective Set Partitioning Problem: {simulation_problem.name}",
        description=simulation_problem.domain_info.problem.description,
        constants=[],
        variables=variables,
        objectives=objectives,
        constraints=constraints,
    )
