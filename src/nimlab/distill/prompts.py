"""Prompt builders for every pipeline call.

Each system message starts with a ``### TASK: <name>`` line. Real models read it as a heading; the
offline simulator uses it to know which kind of sample to produce.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "InstructionList",
    "JudgeScore",
    "QAPair",
    "QuestionList",
    "SubtopicList",
    "TopicList",
    "answer_messages",
    "hard_questions_messages",
    "instructions_messages",
    "judge_messages",
    "qa_messages",
    "reward_messages",
    "subtopics_messages",
    "task_of",
    "topics_messages",
]

Msgs = list[dict[str, Any]]


class TopicList(BaseModel):
    topics: list[str] = Field(min_length=1)


class SubtopicList(BaseModel):
    subtopics: list[str] = Field(min_length=1)


class InstructionList(BaseModel):
    instructions: list[str] = Field(min_length=1)


class QuestionList(BaseModel):
    questions: list[str] = Field(min_length=1)


class QAPair(BaseModel):
    question: str
    answer: str


class JudgeScore(BaseModel):
    score: float = Field(ge=0, le=10)
    rationale: str = ""


def _sys(task: str, body: str) -> dict[str, Any]:
    return {"role": "system", "content": f"### TASK: {task}\n{body}"}


def task_of(messages: Msgs) -> str | None:
    for m in messages:
        content = str(m.get("content") or "")
        if m.get("role") == "system" and content.startswith("### TASK: "):
            return content.split("\n", 1)[0][len("### TASK: ") :].strip()
    return None


def topics_messages(domain: str, k: int) -> Msgs:
    return [
        _sys("topics", "You design broad, non-overlapping curricula. Reply with JSON only."),
        {
            "role": "user",
            "content": f"Domain: {domain}\nList {k} distinct topics inside this domain.",
        },
    ]


def subtopics_messages(topic: str, k: int) -> Msgs:
    return [
        _sys(
            "subtopics", "You split a topic into distinct, concrete angles. Reply with JSON only."
        ),
        {"role": "user", "content": f"Topic: {topic}\nList {k} distinct subtopics."},
    ]


def instructions_messages(topic: str, subtopic: str, k: int) -> Msgs:
    return [
        _sys(
            "instructions",
            "You write realistic, self-contained user requests of varied form and difficulty. "
            "Reply with JSON only.",
        ),
        {
            "role": "user",
            "content": f"Topic: {topic}\nSubtopic: {subtopic}\nWrite {k} different user requests.",
        },
    ]


def hard_questions_messages(topic: str, subtopic: str, k: int) -> Msgs:
    return [
        _sys(
            "hard_questions",
            "You write hard evaluation questions that need multi-step reasoning and have a "
            "defensible reference answer. Reply with JSON only.",
        ),
        {
            "role": "user",
            "content": f"Topic: {topic}\nSubtopic: {subtopic}\nWrite {k} hard questions.",
        },
    ]


def qa_messages(context: str, variant: int, source: str = "") -> Msgs:
    return [
        _sys(
            "qa_pair",
            "Write one question that can be answered from the excerpt alone, and the answer, "
            "quoting or closely paraphrasing the excerpt. Reply with JSON only.",
        ),
        {
            "role": "user",
            "content": f"Source: {source or 'document'}\nVariant: {variant}\n"
            f"Excerpt:\n{context}\n\nWrite question variant {variant}.",
        },
    ]


def answer_messages(prompt: str, system: str | None = None, *, reference: bool = False) -> Msgs:
    task = "reference" if reference else "answer"
    body = (
        "Give a precise, complete, well-reasoned reference answer."
        if reference
        else (system or "You are a helpful, accurate assistant. Answer clearly and concretely.")
    )
    return [_sys(task, body), {"role": "user", "content": prompt}]


def reward_messages(prompt: str, response: str) -> Msgs:
    """The format Nemotron reward models expect: the conversation, ending in the assistant turn."""
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]


def judge_messages(prompt: str, response: str) -> Msgs:
    return [
        _sys(
            "judge",
            "You grade an assistant answer from 0 to 10 on helpfulness, correctness, clarity and "
            "completeness. 0 = useless or harmful, 5 = acceptable, 10 = excellent. "
            "Reply with JSON only.",
        ),
        {"role": "user", "content": f"Question:\n{prompt}\n\nAnswer:\n{response}"},
    ]
