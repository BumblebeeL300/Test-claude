"""Benchmark two models on the bundled suites plus a custom one, then route a prompt.

Runs offline (simulated results) unless NVIDIA_API_KEY is set:

    python examples/arena/quickstart.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from nimlab.arena import Arena, load_suite

CUSTOM = Path(__file__).with_name("custom_suite.json")
MODELS = ["meta/llama-3.3-70b-instruct", "meta/llama-3.1-8b-instruct"]


async def main() -> None:
    async with Arena(concurrency=8) as arena:
        suites = [*arena.resolve_suites("math_reasoning,factual_qa", limit=6), load_suite(CUSTOM)]
        run = await arena.run(MODELS, suites)
        board = arena.leaderboard(run)
        for s in board.standings:
            print(f"#{s.rank} {s.model}: overall {s.overall:.2f}, Elo {s.elo}")
        arena.report(run, html_out="leaderboard.html")
        print(
            "wrote leaderboard.html and", arena.store.path_for(run.run_id) if arena.store else "-"
        )

        decision = await arena.router().route("Solve 3x + 7 = 28")
        print(decision.explain())


if __name__ == "__main__":
    asyncio.run(main())
