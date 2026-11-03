"""Distill quickstart: generate, inspect and export a dataset offline (no API key needed).

    python examples/distill/quickstart.py

Without NVIDIA_API_KEY (or with NIMLAB_MOCK=1) the deterministic simulator stands in for the models,
and every output is labelled as simulated.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from nimlab.distill import Distiller

HERE = Path(__file__).parent


async def main() -> None:
    distiller = Distiller(runs_dir=".nimlab/distill/runs")
    result = await distiller.run(
        "instruction_diversity",
        n=80,
        eval_set=HERE / "eval_holdout.jsonl",
        out="data",
    )
    print(f"run {result.run_id}: kept {len(result.kept)} of {result.meta.n} candidates")
    for row in result.report.funnel:
        print(f"  {row.label:<22} {row.out:>4}  dropped {row.dropped:>3}  {row.reasons}")
    for path in result.files:
        print("wrote", path)
    if result.meta.simulated:
        print("NOTE: simulated output - set NVIDIA_API_KEY to use real models.")


if __name__ == "__main__":
    asyncio.run(main())
