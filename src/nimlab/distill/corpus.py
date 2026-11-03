"""A tiny built-in document set so ``qa_from_docs`` also runs in simulation without ``--docs``."""

from __future__ import annotations

__all__ = ["BUILTIN_DOCS"]

BUILTIN_DOCS: dict[str, str] = {
    "retries.md": """# Retry policies

## Why retries need limits

A retry turns a transient failure into a success, but an unlimited retry loop turns a small outage
into a large one. Every retry policy should therefore have a maximum number of attempts and a
maximum total delay. Without these limits, clients keep hammering a service that is already
struggling.

## Exponential backoff and jitter

Exponential backoff doubles the waiting time after each failed attempt, which gives a recovering
service room to breathe. Jitter adds a random share to every delay so that thousands of clients do
not retry at the same instant. Together, backoff and jitter spread the load of a retry storm over
time and make recovery much faster.

## Idempotency

Only idempotent operations can be retried safely. A payment request that is repeated after a
timeout may charge a customer twice unless the request carries an idempotency key. The server stores
the key and returns the first result for every repeated request.
""",
    "solar.md": """# Home solar power

## How a rooftop system works

Solar panels convert sunlight into direct current. An inverter turns that current into the
alternating current used by household appliances. Any surplus can be stored in a battery or sent to
the grid, depending on the local tariff and the equipment installed.

## Sizing a system

The right size depends on yearly electricity use, roof area and orientation. A south-facing roof
without shade produces the most energy in the northern hemisphere. Installers estimate the yearly
yield in kilowatt-hours per installed kilowatt, and compare it with the household's consumption to
choose the number of panels.

## Maintenance

Panels have no moving parts and need little care. Rain removes most dust, and an annual visual
check of the mounting and the cables is usually enough. Inverters often wear out before the panels
and may need replacement after ten to fifteen years.
""",
    "sourdough.md": """# Sourdough basics

## The starter

A sourdough starter is a living culture of wild yeast and lactic acid bacteria kept in flour and
water. Feeding it regularly keeps the culture active. A healthy starter doubles in size within a few
hours of a feeding and smells pleasantly sour rather than sharp.

## Fermentation

Bulk fermentation develops flavour and structure. Warm kitchens speed it up, while cool
kitchens slow it down, so the dough should be judged by its volume and texture rather than by
the clock. Dough that has grown by about half and feels airy is ready to be shaped.

## Baking

A hot, steamy oven gives the loaf its crisp crust. Baking inside a preheated covered pot traps the
steam released by the dough, which keeps the crust flexible during the first minutes and allows the
loaf to expand fully.
""",
}
