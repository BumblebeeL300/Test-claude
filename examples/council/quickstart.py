"""Ask a council a question with every strategy and save the traces.

    python examples/council/quickstart.py                  # offline mock mode if no NVIDIA_API_KEY
    python examples/council/quickstart.py "Is P = NP?" --out /tmp/council

Writes <out>/<strategy>.json and <out>/<strategy>.html for each strategy.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rich.table import Table

from nimlab.core import NIMClient, get_console
from nimlab.council import Council

STRATEGIES = ("moa", "debate", "best-of-n", "critic")


async def main(question: str, out: Path) -> None:
    console = get_console()
    table = Table(title=f"Council: {question}", title_justify="left", show_lines=True)
    for col in ("Strategy", "Steps", "Tokens", "Agreement", "Time", "Answer"):
        table.add_column(col, overflow="fold")

    async with NIMClient() as client:  # mock backend automatically when no API key is set
        council = Council(client)
        for strategy in STRATEGIES:
            result = await council.run(question, strategy)
            result.save_json(out / f"{strategy}.json")
            result.save_html(out / f"{strategy}.html")
            agreement = f"{result.agreement:.2f}" if result.agreement is not None else "-"
            table.add_row(
                strategy,
                str(len(result.trace.steps)),
                f"{result.trace.total_tokens:,}",
                agreement,
                f"{result.trace.wall_time_s:.2f}s",
                result.answer[:160].replace("\n", " "),
            )
        console.print(table)
        console.print(client.usage.rich_table("Total usage"))
    console.print(f"Traces written to [bold]{out}[/] (open the .html files in a browser).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?", default="Why is the sky blue?")
    parser.add_argument("--out", type=Path, default=Path("council-out"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    asyncio.run(main(args.question, args.out))
