"""Prompt templates. Kept in one place so strategies stay readable and prompts are easy to tune."""

from __future__ import annotations

from collections.abc import Sequence

from .context import Answer, clip
from .schemas import Critique

DEFAULT_SYSTEM = (
    "You are one expert member of a council of AI models answering the same question "
    "independently. Give your best answer: correct, complete and concise. Show the key reasoning, "
    "state assumptions, and say plainly when you are unsure. Do not mention the council."
)


def system_message(text: str | None = None) -> dict[str, str]:
    return {"role": "system", "content": text or DEFAULT_SYSTEM}


def proposer_messages(question: str, system: str | None = None) -> list[dict[str, str]]:
    return [system_message(system), {"role": "user", "content": question}]


def numbered(answers: Sequence[Answer], *, limit: int = 6000) -> str:
    """Anonymised, numbered answers (model names are hidden to avoid brand bias)."""
    return "\n\n".join(
        f"### Response {i}\n{clip(a.text, limit)}" for i, a in enumerate(answers, start=1)
    )


# --------------------------------------------------------------------------- Mixture-of-Agents
MOA_AGGREGATE_SYSTEM = (
    "You have been provided with a set of responses from various AI models to the user's "
    "question. Your task is to synthesise them into a single, high-quality answer. Critically "
    "evaluate the responses: some may be biased, incomplete or wrong. Keep what is well "
    "supported, resolve disagreements by reasoning about which side is correct, and fill gaps. "
    "Do not simply concatenate or list the responses, and do not mention that you were given "
    "responses. Reply with the improved answer only."
)


def moa_messages(
    question: str, previous: Sequence[Answer], system: str | None = None
) -> list[dict[str, str]]:
    user = (
        f"Question:\n{question}\n\nResponses from other models:\n\n{numbered(previous)}\n\n"
        "Write the best possible answer to the question."
    )
    return [
        {"role": "system", "content": system or MOA_AGGREGATE_SYSTEM},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- Debate
DEBATE_CRITIQUE_SYSTEM = (
    "You are a rigorous reviewer in a panel debate. You will see the question and your peers' "
    "answers. For each peer answer, point out concrete factual errors, flawed reasoning, "
    "omissions or unsupported claims, and also note what it gets right. Be specific and brief "
    "(a few bullet points per answer). Finish with one line: 'Overall position:' stating which "
    "answer(s) you find most convincing and why."
)


def debate_critique_messages(question: str, peers: Sequence[Answer]) -> list[dict[str, str]]:
    user = f"Question:\n{question}\n\nPeer answers to review:\n\n{numbered(peers)}"
    return [
        {"role": "system", "content": DEBATE_CRITIQUE_SYSTEM},
        {"role": "user", "content": user},
    ]


DEBATE_REVISE_SYSTEM = (
    "You are a panelist in a debate that is converging on the best answer. Reconsider your "
    "previous answer in light of your peers' answers and their critiques. Concede points where "
    "peers are right, defend points where you are right and explain why, and fix any mistakes. "
    "Reply with your improved, self-contained final answer only."
)


def debate_revise_messages(
    question: str,
    own: Answer,
    peers: Sequence[Answer],
    critiques: Sequence[Answer],
) -> list[dict[str, str]]:
    parts = [f"Question:\n{question}", f"Your previous answer:\n{clip(own.text, 6000)}"]
    if peers:
        parts.append(f"Your peers' answers:\n\n{numbered(peers)}")
    if critiques:
        body = "\n\n".join(
            f"### Critique {i}\n{clip(c.text, 3000)}" for i, c in enumerate(critiques, start=1)
        )
        parts.append(
            f"Critiques written by your peers (they cover all answers, yours included):\n\n{body}"
        )
    parts.append("Write your revised answer.")
    return [
        {"role": "system", "content": DEBATE_REVISE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


DEBATE_JUDGE_SYSTEM = (
    "You are an impartial judge. A panel of AI models debated a question. Read the final "
    "answers (anonymised) and the debate history, decide which answer is best or write a better "
    "synthesis, and justify the decision. Judge on correctness first, then completeness and "
    "clarity - not on length or confidence. Respond with JSON matching the schema: 'selected' is "
    "the 1-based number of the chosen answer (or null if you synthesised), 'final_answer' is the "
    "complete answer to give the user, 'rationale' explains the decision briefly, 'confidence' is "
    "0-1."
)


def debate_judge_messages(
    question: str, finals: Sequence[Answer], history: str
) -> list[dict[str, str]]:
    user = (
        f"Question:\n{question}\n\nFinal answers:\n\n{numbered(finals)}\n\n"
        f"Debate history:\n{history or '(no debate rounds were needed)'}"
    )
    return [{"role": "system", "content": DEBATE_JUDGE_SYSTEM}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- Best-of-N
RUBRIC_SYSTEM = (
    "You are a strict answer grader. Score the candidate answer to the question from 0 to 10 "
    "using this rubric: correctness (most important), completeness, reasoning quality, clarity "
    "and appropriate concision. 0 = wrong or useless, 5 = partially correct or incomplete, 10 = "
    "correct, complete and clear. Respond with JSON: 'score' and a short 'rationale'."
)


def rubric_messages(question: str, answer: str) -> list[dict[str, str]]:
    user = f"Question:\n{question}\n\nCandidate answer:\n{clip(answer, 8000)}"
    return [{"role": "system", "content": RUBRIC_SYSTEM}, {"role": "user", "content": user}]


def reward_messages(question: str, answer: str) -> list[dict[str, str]]:
    """The [user, assistant] pair a NIM reward model scores."""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


# --------------------------------------------------------------------------- Critic loop
DRAFT_SYSTEM = (
    "You are an expert writing a first draft that a reviewer will then critique. Answer the "
    "question fully and correctly; reasoning shown briefly. Reply with the answer only."
)


def draft_messages(question: str, system: str | None = None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system or DRAFT_SYSTEM},
        {"role": "user", "content": question},
    ]


CRITIC_SYSTEM = (
    "You are a meticulous, skeptical reviewer. Find real problems in the draft answer: factual "
    "errors, logical gaps, missing essentials, ambiguity, or unsafe advice. Do not nitpick "
    "style unless it harms understanding. Respond with JSON: 'issues' is a list of objects with "
    "'severity' ('high' = wrong, misleading or missing something essential; 'medium' = notable "
    "weakness; 'low' = minor polish), 'category' (accuracy | completeness | reasoning | clarity "
    "| safety | other), 'description' and 'suggestion'. Use an empty list if the draft needs no "
    "changes. 'summary' is a one or two sentence overall assessment."
)


def critic_messages(question: str, draft: str, iteration: int) -> list[dict[str, str]]:
    user = (
        f"Question:\n{question}\n\nDraft answer (revision {iteration}):\n{clip(draft, 8000)}\n\n"
        "Review the draft."
    )
    return [{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": user}]


REVISER_SYSTEM = (
    "You are revising your own draft answer after a review. Fix every issue the reviewer raised "
    "that is valid (if you disagree with an issue, keep the draft's position but make the "
    "justification explicit). Preserve what was already correct. Reply with the complete revised "
    "answer only - no preamble, no change log."
)


def reviser_messages(question: str, draft: str, critique: Critique) -> list[dict[str, str]]:
    issues = "\n".join(
        f"{i}. [{x.severity.upper()}] ({x.category}) {x.description}"
        + (f"\n   Suggestion: {x.suggestion}" if x.suggestion else "")
        for i, x in enumerate(critique.issues, start=1)
    )
    user = (
        f"Question:\n{question}\n\nCurrent draft:\n{clip(draft, 8000)}\n\n"
        f"Reviewer summary: {critique.summary or '(none)'}\n\nIssues to address:\n"
        f"{issues or '(none)'}"
    )
    return [{"role": "system", "content": REVISER_SYSTEM}, {"role": "user", "content": user}]
