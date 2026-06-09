from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langchain.messages import AIMessage

console = Console()

class UI:
    @staticmethod
    def print_header():
        console.print(Panel.fit(
            "[bold cyan]🤖 ROBOTIC TASK ORCHESTRATOR with Qwen 3 VL 9B[/bold cyan]\n[dim]Ready for instructions...[/dim]",
            border_style="cyan"
        ))

    @staticmethod
    def display_result(messages, elapsed_time=None):
        if isinstance(messages[-1], AIMessage):
            content = messages[-1].content.strip()
            
            # Print the response panel
            console.print(Panel(
                content,
                title="[bold blue]Agent Response[/bold blue]",
                border_style="blue"
            ))
            
            # Print the elapsed time if provided
            if elapsed_time is not None:
                console.print(f"[dim]⚡ Processing time: {elapsed_time:.2f}s[/dim]", justify="right")

    @staticmethod
    def get_input(prompt_text: str):
        return console.input(f"\n[bold yellow]➤ {prompt_text}: [/bold yellow]")

    @staticmethod
    def show_status(text: str):
        return console.status(f"[bold green]{text}[/bold green]")

    @staticmethod
    def print_goodbye():
        console.print(f"\n[bold cyan]👋 Always good to work with you! See you again soon...[/bold cyan]\n")
