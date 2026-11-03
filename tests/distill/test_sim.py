"""The offline simulator: deterministic, varied, and honest about being simulated."""

from __future__ import annotations

import pytest

from nimlab.core import default_registry
from nimlab.distill.prompts import (
    InstructionList,
    JudgeScore,
    QAPair,
    SubtopicList,
    TopicList,
    answer_messages,
    instructions_messages,
    judge_messages,
    qa_messages,
    reward_messages,
    subtopics_messages,
    task_of,
    topics_messages,
)
from nimlab.distill.scoring import parse_reward
from nimlab.distill.sim import SimClient, answer_quality

BIG = "meta/llama-3.3-70b-instruct"
SMALL = "meta/llama-3.1-8b-instruct"
REWARD = default_registry().pick("reward")


async def test_topic_tree_calls_are_coherent() -> None:
    sim = SimClient()
    topics = await sim.chat_json("m", topics_messages("cooking", 4), TopicList)
    assert len(topics.topics) == 4 and len(set(topics.topics)) == 4
    subs = await sim.chat_json("m", subtopics_messages(topics.topics[0], 3), SubtopicList)
    assert all(topics.topics[0] in s for s in subs.subtopics)
    ins = await sim.chat_json(
        "m", instructions_messages(topics.topics[0], subs.subtopics[0], 5), InstructionList
    )
    assert len(ins.instructions) == 5
    assert all(subs.subtopics[0] in p for p in ins.instructions)
    unknown = await sim.chat_json("m", topics_messages("beekeeping", 3), TopicList)
    assert all("beekeeping" in t for t in unknown.topics)


async def test_everything_is_deterministic_and_seed_dependent() -> None:
    msgs = answer_messages("Explain unit testing to a new team member in under 150 words.")
    a = await SimClient(seed=1).chat(BIG, msgs, temperature=0.7)
    b = await SimClient(seed=1).chat(BIG, msgs, temperature=0.7)
    c = await SimClient(seed=2).chat(BIG, msgs, temperature=0.7)
    assert a.content == b.content
    assert a.content != c.content
    assert task_of(msgs) == "answer"


async def test_instruction_lists_contain_deliberate_duplicates() -> None:
    sim = SimClient()
    exact = near = 0
    for i in range(40):
        got = await sim.chat_json("m", instructions_messages("t", f"sub {i}", 12), InstructionList)
        items = got.instructions
        exact += len(items) - len(set(items))
        norm = {" ".join(x.lower().rstrip(" ?.").split()) for x in items}
        near += len(set(items)) - len(norm)
    assert exact > 0 and near >= 0  # duplicates appear; near ones differ only in casing/punctuation


async def test_bigger_models_write_better_answers_and_rewards_follow() -> None:
    sim = SimClient()
    prompts = [f"Write a step-by-step checklist for handling topic number {i}." for i in range(40)]
    scores: dict[str, list[float]] = {BIG: [], SMALL: []}
    for model in (BIG, SMALL):
        for p in prompts:
            r = await sim.chat(model, answer_messages(p), temperature=0.7)
            reply = await sim.chat(REWARD, reward_messages(p, r.content))
            value = parse_reward(reply.content)
            if value is not None:
                scores[model].append(value)
    assert sum(scores[BIG]) / len(scores[BIG]) > sum(scores[SMALL]) / len(scores[SMALL])


async def test_defects_exist_and_reward_penalises_them() -> None:
    sim = SimClient()
    texts = []
    for i in range(300):
        r = await sim.chat(SMALL, answer_messages(f"Explain topic {i} briefly."), temperature=0.9)
        texts.append(r.content)
    assert any("sorry" in t.lower() or "as an ai" in t.lower() for t in texts)  # refusals
    assert any(len(t) < 40 for t in texts)  # fragments
    assert any("guia practica" in t for t in texts)  # non-English
    assert answer_quality("Explain topic", "I'm sorry, but I can't help with that request.") < 0.3
    good = (
        "- Define the goal clearly for the topic.\n- Break the topic into small steps.\n"
        "- Review the results every week and adjust the plan.\n"
        "- Share what you learn with the rest of the team."
    )
    assert answer_quality("Explain the topic goal steps", good) > 0.45


async def test_reward_replies_parse_and_some_are_unparseable() -> None:
    sim = SimClient()
    values = []
    for i in range(300):
        reply = await sim.chat(REWARD, reward_messages(f"q{i}", f"A fair answer number {i}."))
        values.append(parse_reward(reply.content))
    assert any(v is None for v in values) and any(v is not None for v in values)


async def test_qa_pairs_are_grounded_in_the_excerpt_and_judge_returns_a_score() -> None:
    sim = SimClient()
    excerpt = (
        "Exponential backoff doubles the waiting time after each failed attempt. "
        "Jitter adds a random share to every delay so clients do not retry together."
    )
    answers = [
        (await sim.chat_json(BIG, qa_messages(excerpt, v), QAPair, temperature=0.5)).answer
        for v in range(30)
    ]
    assert sum("backoff" in a.lower() or "jitter" in a.lower() for a in answers) >= 20
    judged = await sim.chat_json("m", judge_messages("q", "a decent answer"), JudgeScore)
    assert 0 <= judged.score <= 10


async def test_embeddings_usage_and_failure_injection() -> None:
    sim = SimClient()
    vecs = await sim.embed("e", ["alpha beta", "alpha beta", "gamma delta"])
    assert vecs[0] == vecs[1] != vecs[2]
    assert sim.usage.total_calls == 1
    crashing = SimClient(fail_after=1)
    await crashing.embed("e", ["x"])
    with pytest.raises(RuntimeError, match="simulated crash"):
        await crashing.embed("e", ["x"])
    with pytest.raises(ValueError, match="does not know task"):
        await sim.chat_json("m", [{"role": "user", "content": "hi"}], JudgeScore)
