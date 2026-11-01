"""``code_exec``: run model-written code against hidden unit tests in the sandbox."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base import GradeContext, Grader, GradeResult, register_grader
from .sandbox import extract_code, run_python_tests


class CodeExecParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tests: list[str] = Field(min_length=1)
    entrypoint: str | None = None
    timeout_s: float = Field(default=10.0, gt=0, le=120)
    test_timeout_s: float = Field(default=2.0, gt=0, le=60)
    memory_mb: int = Field(default=512, ge=64, le=4096)

    @field_validator("tests")
    @classmethod
    def _compile_tests(cls, tests: list[str]) -> list[str]:
        for i, src in enumerate(tests):
            try:
                compile(src, f"<hidden test {i}>", "exec")
            except SyntaxError as exc:
                raise ValueError(f"hidden test {i} is not valid Python: {exc}") from exc
        return tests


@register_grader
class CodeExecGrader(Grader):
    """Extract the model's code, run each hidden test in a sandbox; score = fraction passed."""

    name = "code_exec"
    description = "Executes model-written Python against hidden unit tests in a sandbox."
    Params = CodeExecParams

    async def run(self, params: CodeExecParams, ctx: GradeContext) -> GradeResult:
        code = extract_code(ctx.output, params.entrypoint)
        if not code.strip():
            return GradeResult(score=0.0, passed=False, detail="no code found in output")
        res = await run_python_tests(
            code,
            params.tests,
            timeout_s=params.timeout_s,
            test_timeout_s=params.test_timeout_s,
            memory_mb=params.memory_mb,
        )
        return GradeResult(
            score=res.fraction,
            passed=res.ok,
            detail=res.summary(),
            extracted=code[:400],
            meta={"timed_out": res.timed_out, "tests": res.details, "stderr": res.stderr[-300:]},
        )
