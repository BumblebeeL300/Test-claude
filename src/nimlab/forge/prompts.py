"""Prompt templates for each Forge role."""

from __future__ import annotations

from .models import Plan, Review

__all__ = [
    "CODER_SYSTEM",
    "PLANNER_SYSTEM",
    "RCA_SYSTEM",
    "REVIEWER_SYSTEM",
    "fix_prompt",
    "implement_prompt",
    "no_changes_prompt",
    "plan_prompt",
    "rca_prompt",
    "review_fix_prompt",
    "review_prompt",
]

PLANNER_SYSTEM = """\
You are a senior software architect planning work for an autonomous coding agent.
Given a task and a summary of the workspace, produce a short, concrete implementation plan.

Rules:
- Prefer the smallest design that fully solves the task. Python 3 and the standard library only,
  unless the workspace already depends on something else.
- Every step names the files it creates or changes (paths relative to the workspace root).
- The plan must be verifiable with pytest: say which behaviours and edge cases the tests cover.
- List real risks (ambiguous requirements, tricky edge cases), not boilerplate.
Answer with JSON only."""

CODER_SYSTEM = """\
You are Forge, an autonomous coding agent working inside a sandboxed workspace.
You can only act through the provided tools; all paths are relative to the workspace root.

Tools: list_dir, read_file, write_file, apply_patch, run_tests, run_python, search_code.

Working rules:
1. Work test-first: write or extend pytest tests (files named test_*.py) that capture the
   requirements and edge cases, then implement the code, then run run_tests.
2. Use write_file for new files and apply_patch (SEARCH/REPLACE blocks) for edits to existing
   files. Read a file before patching it so the SEARCH text matches exactly.
3. Keep code simple, typed where helpful, documented with short docstrings, and importable from the
   workspace root (put modules next to their tests unless the project already has a layout).
4. Never delete, skip or weaken tests just to make them pass; fix the code instead. Only change
   a test when the test itself is wrong, and say so.
5. No network access, no shell commands, no files outside the workspace.
6. When the tests pass, stop calling tools and reply with a brief summary of what you built."""

REVIEWER_SYSTEM = """\
You are a meticulous senior code reviewer who did NOT write this code.
Review the unified diff for correctness bugs, missed requirements, unhandled edge cases, unsafe
behaviour and weak tests. Ignore pure style preferences unless they hurt readability badly.

Severity: blocker = wrong behaviour or unsafe; major = likely bug or missing requirement;
minor = worth fixing; nit = optional. Choose verdict "request_changes" only when at least one
blocker or major issue exists, otherwise "approve". Be specific: name the file and line.
Answer with JSON only."""

RCA_SYSTEM = """\
You are a principal engineer doing root-cause analysis. An autonomous coding agent keeps
producing the same test failures. Work out WHY from the failures and the source code, then tell the
agent exactly what to change.

Respond in plain text, at most 300 words, in this shape:
Root cause: <one or two sentences>
Fix: <numbered, concrete edits naming files and functions>
Watch out: <anything that would make the same mistake again>"""


def plan_prompt(task: str, workspace_summary: str) -> str:
    return f"## Task\n{task.strip()}\n\n## Workspace\n{workspace_summary}\n\nProduce the plan."


def _render_plan(plan: Plan) -> str:
    lines = [f"Goal: {plan.goal}"]
    for step in plan.steps:
        files = f" [{', '.join(step.files)}]" if step.files else ""
        lines.append(f"{step.id}. {step.description}{files}")
    if plan.test_strategy:
        lines.append(f"Test strategy: {plan.test_strategy}")
    if plan.risks:
        lines.append("Risks: " + "; ".join(plan.risks))
    return "\n".join(lines)


def implement_prompt(task: str, plan: Plan | None, workspace_summary: str) -> str:
    plan_text = _render_plan(plan) if plan else "(no plan available; use your judgement)"
    return (
        f"## Task\n{task.strip()}\n\n## Plan\n{plan_text}\n\n## Workspace\n{workspace_summary}\n\n"
        "Implement the task now. Write the tests together with the code, run run_tests, and fix "
        "what fails before you finish."
    )


def fix_prompt(report: str, iteration: int, max_iters: int, rca: str | None = None) -> str:
    parts = [
        f"The test suite still fails (fix round {iteration} of at most {max_iters}).",
        "",
        report,
    ]
    if rca:
        parts += ["", "A senior engineer analysed the repeated failure:", rca.strip()]
    parts += [
        "",
        "Find the cause and fix the code (read the relevant files first if you are unsure). "
        "Do not weaken or delete tests. Run run_tests to confirm, then reply with a short summary.",
    ]
    return "\n".join(parts)


def no_changes_prompt(iteration: int, max_iters: int) -> str:
    return (
        f"Round {iteration} of at most {max_iters}: the tests pass, but no file in the workspace "
        "has been created or modified, so the task is NOT implemented yet. Use the tools: "
        "write the tests for the requested behaviour, implement it, and run run_tests. Do not "
        "just describe what you would do."
    )


def rca_prompt(
    task: str,
    plan: Plan | None,
    report: str,
    history: list[str],
    sources: dict[str, str],
    coder_notes: str,
) -> str:
    parts = [f"## Task\n{task.strip()}"]
    if plan:
        parts.append(f"## Plan\n{_render_plan(plan)}")
    parts.append(f"## Repeating failures\n{report}")
    if history:
        parts.append("## Earlier test rounds\n" + "\n".join(f"- {h}" for h in history))
    if coder_notes.strip():
        parts.append(f"## What the coder said it did last\n{coder_notes.strip()[:1500]}")
    for path, text in sources.items():
        parts.append(f"## {path}\n```python\n{text}\n```")
    return "\n\n".join(parts)


def review_prompt(task: str, plan: Plan | None, tests_summary: str, diff: str) -> str:
    plan_text = _render_plan(plan) if plan else "(none)"
    return (
        f"## Task\n{task.strip()}\n\n## Plan\n{plan_text}\n\n## Test status\n{tests_summary}\n\n"
        f"## Unified diff\n```diff\n{diff}\n```\n\nReview this change."
    )


def review_fix_prompt(review: Review) -> str:
    lines = ["An independent reviewer requested changes:", "", review.summary.strip(), ""]
    for issue in review.issues:
        loc = f" ({issue.file}{':' + str(issue.line) if issue.line else ''})" if issue.file else ""
        fix = f" Suggested fix: {issue.suggestion}" if issue.suggestion else ""
        lines.append(f"- [{issue.severity}]{loc} {issue.description}{fix}")
    lines += [
        "",
        "Address the blocker and major issues. Add or adjust tests for them, keep every existing "
        "test passing, run run_tests, then reply with a short summary.",
    ]
    return "\n".join(lines)
