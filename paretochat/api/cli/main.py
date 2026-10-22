"""Main CLI module to run ParetoChat examples."""

from typing import Annotated, Optional

import typer

from paretochat.examples.biobj_discrete_choice.main import main as run_biobj_discrete_choice
from paretochat.examples.courier_order_assignment.main import main as run_courier_order_assignment
from paretochat.examples.three_obj_knapsack.main import main as run_three_obj_knapsack

app = typer.Typer()

AutoApprove = Annotated[
    bool,
    typer.Option(help="Run the algorithm without asking for approval. Use --no-auto-approve to confirm each run."),
]


@app.command()
def biobj_discrete_choice(auto_approve: AutoApprove = True):
    """Run the biobj discrete choice example."""
    run_biobj_discrete_choice(auto_approve=auto_approve)


@app.command()
def courier_order_assignment(
    instance_dir: Annotated[
        str, typer.Option(help="Path to the directory containing courier instance JSON files.")
    ] = "instance",
    output_csv: Annotated[str, typer.Option(help="Path for the solver output CSV file.")] = "run.csv",
    jar_path: Annotated[Optional[str], typer.Option(help="Path to the solver JAR.")] = None,
    auto_approve: AutoApprove = True,
):
    """Run the courier order assignment example."""
    run_courier_order_assignment(
        instance_dir=instance_dir, output_csv=output_csv, jar_path=jar_path, auto_approve=auto_approve
    )


@app.command()
def three_obj_knapsack(auto_approve: AutoApprove = True):
    """Run the three objective knapsack example."""
    run_three_obj_knapsack(auto_approve=auto_approve)


if __name__ == "__main__":
    app()
