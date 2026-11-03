"""Watch a debate happen: live terminal view, early stopping and a custom panel.

    NIMLAB_MOCK_LATENCY=0.3 python examples/council/live_debate.py

With a real key, set NVIDIA_API_KEY and drop NIMLAB_MOCK_LATENCY. The panel below can be any NIM
chat models; the judge defaults to the first one.
"""

from __future__ import annotations

import asyncio
import os

from nimlab.core import NIMClient
from nimlab.council import Council, CouncilLive, StepType

PANEL = [
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-super-120b-a12b",
    "google/gemma-4-31b-it",
    "mistralai/mistral-large-4-1t-a52b",
]


async def main() -> None:
    question = "Should a startup use microservices from day one? Answer in a paragraph."
    os.environ.setdefault("NIMLAB_MOCK_LATENCY", "0.15")  # only affects the offline mock backend
    async with NIMClient() as client:
        with CouncilLive() as view:
            result = await Council(client).run(
                question,
                "debate",
                models=PANEL,
                rounds=3,
                convergence_threshold=0.85,  # stop as soon as the panel (mostly) agrees
                observers=[view],
            )

    meta = result.trace.meta
    print(f"\nAnswer:\n{result.answer}\n")
    print(f"Judge rationale: {result.rationale}")
    print(
        f"Rounds run: {meta['rounds_run']}/{meta['rounds_max']} "
        f"(stopped early: {meta['stopped_early']}); "
        f"agreement by round: {meta['agreement_by_round']}"
    )
    final = result.consensus
    if final and final.outliers:
        print(f"Outlier answers: {', '.join(final.outlier_labels)}")
    critiques = len(result.trace.steps_of(StepType.CRITIQUE))
    steps, tokens = len(result.trace.steps), result.trace.total_tokens
    print(f"{steps} steps, {critiques} critiques, {tokens:,} tokens")

    path = result.save_html("debate-trace.html")
    result.save_json("debate-trace.json")
    print(f"Open {path} in a browser for the timeline, consensus heatmaps and every step.")


if __name__ == "__main__":
    asyncio.run(main())
