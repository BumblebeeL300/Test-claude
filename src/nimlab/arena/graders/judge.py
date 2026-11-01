"""``llm_judge``: a judge model scores the answer 1-10 against a rubric."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nimlab.core.errors import NIMError

from .base import GradeContext, Grader, GraderError, GradeResult, register_grader

__all__ = ["JUDGE_SYSTEM", "JudgeVerdict", "build_judge_messages"]

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator. Score the candidate response to the task using only "
    "the rubric. Ignore the response's length, tone and formatting unless the rubric mentions "
    "them. Use the full 1-10 scale: 1 = useless or wrong, 5 = partially meets the rubric, "
    "10 = fully meets every criterion. Reply with JSON only."
)


class JudgeVerdict(BaseModel):
    """What the judge must return."""

    score: int = Field(ge=1, le=10, description="1 (worst) to 10 (best)")
    rationale: str = Field(description="One or two sentences justifying the score")


def build_judge_messages(
    *, prompt: str, response: str, rubric: str, reference: str | None
) -> list[dict[str, Any]]:
    parts = [f"## Task given to the model\n{prompt.strip()}", f"## Rubric\n{rubric.strip()}"]
    if reference:
        parts.append(f"## Reference answer (a good answer, not the only one)\n{reference.strip()}")
    parts.append(f"## Candidate response\n{response.strip() or '(empty response)'}")
    parts.append('Return JSON: {"score": <integer 1-10>, "rationale": "<why>"}')
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


class JudgeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rubric: str = Field(min_length=3)
    pass_threshold: float = Field(default=0.7, ge=0, le=1)
    use_reference: bool = True
    judge_model: str | None = None


@register_grader
class LLMJudgeGrader(Grader):
    """Score = ``(judge_score - 1) / 9``; passed when it reaches ``pass_threshold``."""

    name = "llm_judge"
    description = "A judge model scores the response 1-10 against a rubric."
    needs_llm = True
    Params = JudgeParams

    async def run(self, params: JudgeParams, ctx: GradeContext) -> GradeResult:
        model = params.judge_model or ctx.judge_model
        if ctx.client is None or not model:
            raise GraderError("llm_judge needs an LLM client and a judge model")
        messages = build_judge_messages(
            prompt=ctx.prompt,
            response=ctx.output,
            rubric=params.rubric,
            reference=ctx.reference if params.use_reference else None,
        )
        try:
            verdict = await ctx.client.chat_json(
                model, messages, JudgeVerdict, temperature=0.0, max_tokens=400
            )
        except NIMError as exc:
            raise GraderError(f"judge call failed: {exc}") from exc
        score = (verdict.score - 1) / 9
        return GradeResult(
            score=score,
            passed=score >= params.pass_threshold,
            detail=f"judge {verdict.score}/10: {verdict.rationale.strip()[:300]}",
            meta={"judge_model": model, "judge_score": verdict.score},
        )
