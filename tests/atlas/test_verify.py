"""Claim splitting, scripted LLM verification, lexical fallback and revision."""

from __future__ import annotations

import pytest

from nimlab.atlas import Atlas
from nimlab.atlas.models import SourceRef
from nimlab.atlas.verify import (
    Claim,
    ClaimJudgement,
    Verifier,
    build_judge_messages,
    build_revision_messages,
    lexical_judge,
    normalize_verdict,
    quote_in_sources,
    split_claims,
    split_sentences,
    summarize,
)
from nimlab.core import NIMAuthError, NIMJSONError

from .conftest import FakeClient

SOURCES = [
    SourceRef(
        n=1,
        chunk_id=1,
        source="architecture.md",
        path="/a",
        text="Robots send a heartbeat every 2 seconds. A robot is lost after 3 missed heartbeats.",
    ),
    SourceRef(
        n=2,
        chunk_id=2,
        source="api_guide.md",
        path="/b",
        text="Rotate API keys every 90 days. Send the key in the X-Hive-Key header.",
    ),
]


# ----------------------------------------------------------------------------- splitting


def test_claims_are_one_per_sentence_with_their_citations():
    claims = split_claims(
        "Robots send heartbeats every 2 seconds [1]. "
        "A robot is lost after three missed heartbeats [1]. "
        "Keys rotate every 90 days [2]."
    )
    assert [c.cited for c in claims] == [[1], [1], [2]]
    assert claims[0].plain == "Robots send heartbeats every 2 seconds."
    assert [c.id for c in claims] == [1, 2, 3]


def test_citation_after_the_period_attaches_to_the_previous_sentence():
    claims = split_claims(
        "The key header is X-Hive-Key. [2] Keys rotate every 90 days. [1][2] Done soon today."
    )
    assert claims[0].cited == [2] and claims[1].cited == [1, 2] and claims[2].cited == []
    assert claims[0].plain == "The key header is X-Hive-Key."


def test_bullets_and_numbered_lists_become_separate_claims():
    text = (
        "Steps:\n- Restart the allocator pods now [1]\n"
        "- Wait ninety seconds for recovery [1]\n1. Check the queue depth metric [1]"
    )
    claims = split_claims(text)
    assert len(claims) == 3 and all(c.cited == [1] for c in claims)
    assert claims[0].plain == "Restart the allocator pods now"


def test_headings_questions_leadins_and_abstentions_are_not_claims():
    text = (
        "## Summary\n\nHere is what I found:\nWhat about the rest?\n"
        "The provided sources do not cover pricing.\nRobots park themselves when hot [1]."
    )
    assert [c.plain for c in split_claims(text)] == ["Robots park themselves when hot."]


def test_abbreviations_decimals_and_code_do_not_split_sentences():
    text = (
        "Use tools e.g. kubectl or curl for debugging [1]. "
        "Version 2.5 ships with `a.b.c()` support [1]."
    )
    claims = split_claims(text)
    assert len(claims) == 2
    assert (
        "e.g. kubectl" in claims[0].text
        and "2.5" in claims[1].text
        and "`a.b.c()`" in claims[1].text
    )
    assert split_sentences("Dr. Smith arrived. He sat.") == ["Dr. Smith arrived.", "He sat."]


def test_meta_sentences_without_citations_are_skipped_but_cited_ones_kept():
    assert split_claims("Overall, this is great news for everyone.") == []
    assert len(split_claims("Overall, the budget is 250 EUR per rotation [1].")) == 1


def test_fenced_code_blocks_are_not_claims():
    text = "Run this to watch heartbeats [1].\n\n```\nnats sub robot.*.heartbeat now\n```\n"
    assert [c.plain for c in split_claims(text)] == ["Run this to watch heartbeats."]


def test_short_fragments_are_ignored():
    assert split_claims("Yes. OK then.") == []


# ----------------------------------------------------------------------------- judging helpers


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("supported", "supported"),
        ("Supported", "supported"),
        ("fully_supported", "supported"),
        ("partially_supported", "partial"),
        ("partial", "partial"),
        ("unsupported", "unsupported"),
        ("not supported", "unsupported"),
        ("contradicted", "unsupported"),
        ("cannot be verified", "unsupported"),
        ("no", "unsupported"),
    ],
)
def test_verdict_normalisation(raw: str, expected: str):
    assert normalize_verdict(raw) == expected


def test_garbage_verdicts_are_rejected_so_chat_json_retries():
    with pytest.raises(ValueError):
        normalize_verdict("banana")
    with pytest.raises(ValueError):
        ClaimJudgement(verdict="banana")


def test_lexical_judge_thresholds_and_evidence():
    passages = [SOURCES[0].text]
    verdict, evidence, cov = lexical_judge("Robots send a heartbeat every 2 seconds", passages)
    assert verdict == "supported" and cov >= 0.8 and "heartbeat" in evidence
    assert (
        lexical_judge(
            "A robot is lost after missed heartbeats plus unrelated pricing tiers", passages
        )[0]
        == "partial"
    )
    assert lexical_judge("Quarterly revenue grew sharply in Europe", passages)[0] == "unsupported"
    assert lexical_judge("anything", []) == ("unsupported", "", 0.0)


def test_quote_matching_is_forgiving_about_case_and_punctuation():
    passages = [SOURCES[1].text]
    assert quote_in_sources("rotate api keys every 90 days", passages)
    assert quote_in_sources('Send the key in the "X-Hive-Key" header', passages)
    assert not quote_in_sources("Keys never expire", passages)
    assert not quote_in_sources("", passages)


def test_summarize_scores_supported_partial_unsupported():
    from nimlab.atlas.models import ClaimCheck

    checks = [
        ClaimCheck(id=1, text="a", verdict="supported"),
        ClaimCheck(id=2, text="b", verdict="partial"),
        ClaimCheck(id=3, text="c", verdict="unsupported"),
        ClaimCheck(id=4, text="d", verdict="supported"),
    ]
    report = summarize(checks, method="llm")
    assert report.groundedness == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    assert (report.supported, report.partial, report.unsupported) == (2, 1, 1)
    assert [c.id for c in report.unsupported_claims] == [3]
    assert [c.id for c in report.weak_claims] == [2, 3]
    assert summarize([], method="llm").groundedness is None


# ----------------------------------------------------------------------------- LLM verifier


def judge_script(verdicts: dict[str, dict]):
    """chat_json script: pick the verdict whose key occurs in the CLAIM block of the prompt."""

    def handler(messages):
        claim = messages[-1]["content"].split("CLAIM\n", 1)[1]
        for key, value in verdicts.items():
            if key in claim:
                return value
        return {"verdict": "unsupported", "evidence": "", "reason": "default"}

    return handler


async def test_verifier_maps_scripted_judgements_to_a_report():
    client = FakeClient(
        json={
            "ClaimJudgement": judge_script(
                {
                    "heartbeat every 2 seconds": {
                        "verdict": "supported",
                        "evidence": "Robots send a heartbeat every 2 seconds.",
                        "reason": "stated",
                    },
                    "lost after three": {
                        "verdict": "partially_supported",
                        "evidence": "",
                        "reason": "count differs",
                    },
                    "rotate every 30 days": {
                        "verdict": "unsupported",
                        "evidence": "",
                        "reason": "says 90",
                    },
                }
            )
        }
    )
    verifier = Verifier(client, "judge-model")
    answer = (
        "Robots send a heartbeat every 2 seconds [1]. "
        "A robot is lost after three missed beats [1]. "
        "Keys rotate every 30 days [2]."
    )
    report = await verifier.verify(answer, SOURCES)
    assert [c.verdict for c in report.claims] == ["supported", "partial", "unsupported"]
    assert report.groundedness == pytest.approx(0.5)
    assert report.method == "llm" and not report.truncated
    assert report.claims[0].evidence_found is True
    assert report.claims[0].cited == [1] and not report.claims[0].uncited
    assert {c["model"] for c in client.json_calls} == {"judge-model"}


async def test_each_claim_is_judged_only_against_its_cited_sources():
    client = FakeClient(json={"ClaimJudgement": {"verdict": "supported"}})
    await Verifier(client, "m").verify("Keys rotate every 90 days [2].", SOURCES)
    prompt = client.json_calls[0]["messages"][-1]["content"]
    assert "Rotate API keys every 90 days" in prompt
    assert "heartbeat" not in prompt  # source 1 was not cited


async def test_uncited_claims_are_flagged_and_checked_against_the_best_matching_sources():
    client = FakeClient(json={"ClaimJudgement": {"verdict": "supported"}})
    report = await Verifier(client, "m").verify("Keys rotate every 90 days.", SOURCES)
    assert report.claims[0].uncited is True and report.claims[0].cited == []
    assert "Rotate API keys" in client.json_calls[0]["messages"][-1]["content"]


async def test_fabricated_quote_downgrades_a_supported_verdict():
    client = FakeClient(
        json={
            "ClaimJudgement": {
                "verdict": "supported",
                "evidence": "Keys never need to be rotated at all.",
                "reason": "sure",
            }
        }
    )
    report = await Verifier(client, "m").verify("Keys rotate every 90 days [2].", SOURCES)
    claim = report.claims[0]
    assert claim.verdict == "partial" and claim.evidence_found is False
    assert "quoted evidence not found" in claim.reason


async def test_judge_failure_falls_back_to_the_lexical_judge_per_claim():
    client = FakeClient(json={"ClaimJudgement": NIMJSONError("bad json", raw="???")})
    report = await Verifier(client, "m").verify("Rotate API keys every 90 days [2].", SOURCES)
    assert report.claims[0].method == "lexical-fallback" and report.claims[0].verdict == "supported"
    assert report.method == "mixed"


async def test_auth_errors_propagate_from_the_judge():
    client = FakeClient(json={"ClaimJudgement": NIMAuthError("nope")})
    with pytest.raises(NIMAuthError):
        await Verifier(client, "m").verify("Keys rotate every 90 days [2].", SOURCES)


async def test_lexical_only_mode_makes_no_model_calls():
    client = FakeClient()
    report = await Verifier(client, "m", lexical_only=True).verify(
        "Rotate API keys every 90 days [2]. Pricing doubled across Europe last quarter [1].",
        SOURCES,
    )
    assert client.json_calls == [] and report.method == "lexical"
    assert [c.verdict for c in report.claims] == ["supported", "unsupported"]


async def test_claim_limit_truncates_and_flags():
    client = FakeClient(json={"ClaimJudgement": {"verdict": "supported"}})
    answer = " ".join(f"Statement number {i} says robots are great today [1]." for i in range(10))
    report = await Verifier(client, "m", max_claims=4).verify(answer, SOURCES)
    assert len(report.claims) == 4 and report.truncated and len(client.json_calls) == 4


async def test_answer_without_checkable_claims_has_no_score():
    report = await Verifier(FakeClient(), "m").verify(
        "The provided sources do not cover this.", SOURCES
    )
    assert report.claims == [] and report.groundedness is None


def test_judge_and_revision_prompts_contain_the_essentials():
    claim = Claim(id=1, text="X [1]", plain="Robots rotate keys", cited=[1])
    msgs = build_judge_messages(claim, SOURCES[:1])
    assert (
        "CLAIM\nRobots rotate keys" in msgs[1]["content"]
        and "[1] architecture.md" in msgs[1]["content"]
    )
    from nimlab.atlas.models import ClaimCheck, VerificationReport

    report = VerificationReport(
        claims=[
            ClaimCheck(id=1, text="Bad claim [1]", verdict="unsupported", reason="not in sources")
        ],
        unsupported=1,
    )
    revision = build_revision_messages("Q?", "Draft text", report, SOURCES)
    assert "Bad claim" in revision[1]["content"] and "not in sources" in revision[1]["content"]
    assert (
        "Draft text" in revision[1]["content"]
        and "revising a draft answer" in revision[0]["content"]
    )


# ----------------------------------------------------------------------------- end-to-end revise


def pipeline_client(first: str, revised: str, verdict_for: dict[str, str]) -> FakeClient:
    def chat(model, messages):
        system = messages[0]["content"]
        return revised if "revising a draft answer" in system else first

    def judge(messages):
        claim = messages[-1]["content"].split("CLAIM\n", 1)[1]
        for key, verdict in verdict_for.items():
            if key in claim:
                return {"verdict": verdict, "evidence": "", "reason": "scripted"}
        return {"verdict": "supported", "evidence": "", "reason": "scripted"}

    return FakeClient(chat=chat, json={"ClaimJudgement": judge})


async def test_auto_revision_replaces_unsupported_claims(db_path, corpus_dir):
    client = pipeline_client(
        "Robots send heartbeats every 2 seconds [1]. Robots also run on solar power [1].",
        "Robots send heartbeats every 2 seconds [1].",
        {"solar power": "unsupported"},
    )
    atlas = Atlas(client, db_path)
    await atlas.ingest(corpus_dir)
    plain = await atlas.ask("How often do robots send heartbeats?", verify=True)
    assert plain.verification and plain.verification.unsupported == 1
    assert plain.verification.groundedness == pytest.approx(0.5) and not plain.verification.revised
    revised = await atlas.ask("How often do robots send heartbeats?", revise=True)
    v = revised.verification
    assert v is not None and v.revised and v.unsupported == 0 and v.groundedness == 1.0
    assert v.groundedness_before_revision == pytest.approx(0.5)
    assert revised.text == "Robots send heartbeats every 2 seconds [1]."
    assert revised.original_text and "solar power" in revised.original_text
    assert [s.kind for s in revised.trace if s.kind == "revise"]
    atlas.close()


async def test_revision_that_does_not_improve_is_discarded(db_path, corpus_dir):
    client = pipeline_client(
        "Robots send heartbeats every 2 seconds [1]. Robots also run on solar power [1].",
        "Robots run on solar power and wind power [1].",
        {"solar power": "unsupported", "wind power": "unsupported"},
    )
    atlas = Atlas(client, db_path)
    await atlas.ingest(corpus_dir)
    answer = await atlas.ask("How often do robots send heartbeats?", revise=True)
    assert answer.verification is not None and not answer.verification.revised
    assert answer.original_text is None and "solar power [1]." in answer.text
    atlas.close()
