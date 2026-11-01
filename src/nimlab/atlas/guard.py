"""Safety guard around NVIDIA's NemoGuard content-safety model.

The model is prompted with its documented instruction template and answers with JSON-ish text such
as ``{"User Safety": "unsafe", "Safety Categories": "Violence, Criminal Planning"}``. Real-world
replies are messy (fences, prose, truncated JSON, Llama-Guard style ``unsafe\\nS1``), so parsing is
deliberately forgiving.
"""

from __future__ import annotations

import re
from typing import Any

from nimlab.core import NIMAuthError, NIMError, extract_json

from .compat import is_mock
from .models import GuardVerdict, Stage

CATEGORIES: dict[str, str] = {
    "S1": "Violence",
    "S2": "Sexual",
    "S3": "Criminal Planning/Confessed Criminal Activity",
    "S4": "Guns and Illegal Weapons",
    "S5": "Controlled/Regulated Substances",
    "S6": "Suicide and Self Harm",
    "S7": "Sexual (minor)",
    "S8": "Hate/Identity Hate",
    "S9": "PII/Privacy",
    "S10": "Harassment",
    "S11": "Threat",
    "S12": "Profanity",
    "S13": "Needs Caution",
    "S14": "Other",
    "S15": "Manipulation",
    "S16": "Fraud/Deception",
    "S17": "Malware",
    "S18": "High Risk Gov Decision Making",
    "S19": "Political/Misinformation/Conspiracy",
    "S20": "Copyright/Trademark/Plagiarism",
    "S21": "Unauthorized Advice",
    "S22": "Illegal Activity",
    "S23": "Immoral/Unethical",
}

_TEMPLATE = """Task: Check if there is unsafe content in the user or agent messages in the below \
conversation in accordance with our safety policy with the below categories. Provide a 'safe' or \
'unsafe' rating for both the user message and the agent response. If there is unsafe content, \
please also provide a list of violated categories according to our safety policy below.

<BEGIN UNSAFE CONTENT CATEGORIES>
{categories}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

user: {user}
{response}
<END CONVERSATION>

Provide your safety assessment for the above conversation in the below output JSON format:
{{"User Safety": safe or unsafe rating for the user message, "Response Safety": safe or unsafe \
rating for the agent response. Omit if no agent response present. "Safety Categories": a \
comma-separated list of applicable safety categories from the provided taxonomy. Omit if all \
safe.}}

Do not include anything other than the output JSON in your response.
Output JSON:"""

REFUSAL_QUERY = (
    "I can't help with that request because it was flagged by the safety filter"
    "{cats}. Please rephrase your question."
)
REFUSAL_ANSWER = (
    "The generated answer was withheld because it was flagged by the safety filter{cats}."
)


def build_guard_messages(user_text: str, response_text: str | None = None) -> list[dict[str, Any]]:
    """Single user message in the format NemoGuard content-safety was trained on."""
    categories = "\n".join(f"{code}: {name}." for code, name in CATEGORIES.items())
    response = f"\nresponse: agent: {response_text}\n" if response_text is not None else ""
    prompt = _TEMPLATE.format(categories=categories, user=user_text, response=response)
    return [{"role": "user", "content": prompt}]


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z]", "", key.lower())


def _rating(value: Any) -> bool | None:
    """``True`` = safe, ``False`` = unsafe, ``None`` = unclear."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if re.search(r"\b(unsafe|harmful|violat\w*|flagged)\b", text):
        return False
    if re.search(r"\b(safe|unharmful|harmless|benign|ok|allowed)\b", text):
        return True
    return None


def _split_categories(value: Any) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else re.split(r",|;|\n", str(value))
    out: list[str] = []
    for item in items:
        name = str(item).strip().strip(".")
        if not name:
            continue
        code = re.fullmatch(r"S(\d{1,2})", name, re.IGNORECASE)
        name = CATEGORIES.get(f"S{code.group(1)}", name) if code else name
        if name.lower() not in {"none", "n/a", "safe"} and name not in out:
            out.append(name)
    return out


def parse_guard_output(raw: str, stage: Stage = "query") -> tuple[bool | None, list[str]]:
    """Parse a guard reply into ``(safe, categories)``; ``safe`` is ``None`` when unparseable.

    For ``stage="answer"`` the *response* rating is used (falling back to the user rating when the
    reply carries only one).
    """
    data: Any = None
    try:
        data = extract_json(raw)
    except ValueError:
        data = None
    if isinstance(data, dict):
        norm = {_norm_key(k): v for k, v in data.items()}
        order = ["responsesafety", "usersafety"] if stage == "answer" else ["usersafety"]
        order += ["safety", "rating", "label", "verdict", "result"]
        for key in order:
            if key in norm:
                rating = _rating(norm[key])
                if rating is not None:
                    cats = _split_categories(
                        norm.get("safetycategories") or norm.get("categories") or norm.get("violations")
                    )
                    return rating, ([] if rating else cats)
    # Fallbacks for non-JSON text.
    text = raw or ""
    pattern = r"(user|response)\s*safety[\"'\s:=]*\b(safe|unsafe)\b"
    ratings = {m.group(1).lower(): m.group(2).lower() for m in re.finditer(pattern, text, re.I)}
    want = "response" if stage == "answer" and "response" in ratings else "user"
    chosen: bool | None
    if want in ratings:
        chosen = ratings[want] == "safe"
    else:
        head = text.strip().lower()[:200]
        chosen = _rating(head) if re.match(r"\W*(un)?safe\b|\W*(harmful|unharmful)\b", head) else None
    codes = re.findall(r"\bS\d{1,2}\b", text)
    named = re.search(r"categories[\"'\s:=]+([^\n}]+)", text, re.IGNORECASE)
    cats = _split_categories(",".join(codes) if codes else (named.group(1) if named else None))
    if chosen is None:
        return None, []
    return chosen, ([] if chosen else cats)


class SafetyGuard:
    """Screens queries and answers with the safety model."""

    def __init__(self, client: Any, model: str, *, fail_open: bool | None = None) -> None:
        self.client = client
        self.model = model
        # Unparseable replies are allowed through only when configured (default: only in mock).
        self.fail_open = is_mock(client) if fail_open is None else fail_open

    async def check(
        self, stage: Stage, user_text: str, response_text: str | None = None
    ) -> GuardVerdict:
        if is_mock(self.client):
            return GuardVerdict(stage=stage, safe=True, skipped=True, raw="mock mode: not screened")
        try:
            result = await self.client.chat(
                self.model,
                build_guard_messages(user_text, response_text),
                temperature=0.0,
                max_tokens=96,
            )
        except NIMAuthError:
            raise
        except NIMError as exc:
            return GuardVerdict(
                stage=stage,
                safe=self.fail_open,
                uncertain=True,
                raw=f"guard call failed: {exc}",
            )
        safe, categories = parse_guard_output(result.content, stage)
        if safe is None:
            return GuardVerdict(
                stage=stage,
                safe=self.fail_open,
                uncertain=True,
                raw=result.content,
            )
        return GuardVerdict(
            stage=stage,
            safe=safe,
            categories=categories,
            raw=result.content,
        )


def refusal_text(stage: Stage, categories: list[str]) -> str:
    cats = f" ({', '.join(categories)})" if categories else ""
    template = REFUSAL_QUERY if stage == "query" else REFUSAL_ANSWER
    return template.format(cats=cats)
