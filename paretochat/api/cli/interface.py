"""ParetoChat CLI Interface
Provides an interactive command-line interface for users to interact with the ParetoChatAgent."""

from typing import Callable

from langchain_core.messages import AIMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from paretochat.domain.agents.agent import ParetoChatAgent, approve_all


def confirm_run(request: dict) -> bool:
    """Ask the user to approve running the algorithm with the proposed scalarization."""
    console = Console()
    scalarization = request["objective_scalarization"]
    console.print(Panel(scalarization.model_dump_json(indent=2), title="Proposed scalarization", border_style="yellow"))
    return Confirm.ask("Run the algorithm with this scalarization?", default=True)


def get_approval_handler(auto_approve: bool) -> Callable[[dict], bool]:
    """Return the approval handler for the given mode: auto-approve or ask the user."""
    return approve_all if auto_approve else confirm_run


def interactive_main(agent: ParetoChatAgent, example_queries: list[str] | None = None):
    """Interactive main function for continuous interaction with ParetoChatAgent."""
    console = Console()

    # Welcome message
    welcome_text = """
# ParetoChat

**Multi-Objective Decision Support System**

Welcome! I can help you find your preferred solutions to your multi-objective problem.

### Example queries:
"""
    if example_queries:
        welcome_text += "\n"
        for query in example_queries:
            welcome_text += f"- *{query}*\n"
    welcome_text += "\nType `quit`, `exit`, or `q` to exit.\n"

    console.print(Panel(Markdown(welcome_text), border_style="cyan"))
    console.print()

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold blue]You[/bold blue]")

            # Check for exit commands
            if user_input.lower() in ("quit", "exit", "q", "bye"):
                console.print("\n[bold green]Thank you for using ParetoChat. Goodbye![/bold green]\n")
                break

            # Skip empty input
            if not user_input:
                console.print("[yellow]Please enter a command or question.[/yellow]")
                continue

            # Run the agent
            console.print("\n[dim]Processing...[/dim]\n")
            result = agent.run(user_input)

            # Display the response with markdown rendering
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if isinstance(last_message, AIMessage):
                    console.print("[bold green]ParetoChat:[/bold green]")
                    console.print(Markdown(last_message.content))
                    console.print()
            else:
                console.print("[yellow]No response generated.[/yellow]\n")

        except KeyboardInterrupt:
            console.print("\n\n[bold green]Session interrupted. Goodbye![/bold green]\n")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}\n")
            console.print("Please try again with a different query.\n")
