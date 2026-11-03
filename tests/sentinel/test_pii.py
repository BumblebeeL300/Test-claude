"""Luhn / IBAN checks, the PII regexes and reversible redaction."""

# ruff: noqa: E501

from __future__ import annotations

import pytest

from nimlab.sentinel.pii import (
    PII_KINDS,
    Redactor,
    find_pii,
    iban_valid,
    luhn_valid,
    redact,
    restore,
)

FAKE_KEY = "nvapi-FAKEFAKEFAKEFAKEFAKEFAKE0001"


def kinds(text: str) -> list[str]:
    return [m.kind for m in find_pii(text)]


@pytest.mark.parametrize(
    "number",
    [
        "4111 1111 1111 1111",
        "4111111111111111",
        "5500-0000-0000-0004",
        "378282246310005",
        "6011111111111117",
    ],
)
def test_luhn_valid(number: str) -> None:
    assert luhn_valid(number)


@pytest.mark.parametrize(
    "number", ["4111 1111 1111 1112", "1234567890123456", "123", "4111 1111 1111 11ab", ""]
)
def test_luhn_invalid(number: str) -> None:
    assert not luhn_valid(number)


def test_iban_checksum() -> None:
    assert iban_valid("GB82 WEST 1234 5698 7654 32")
    assert iban_valid("DE89370400440532013000")
    assert not iban_valid("GB82 WEST 1234 5698 7654 33")
    assert not iban_valid("XX00")


def test_email_detection() -> None:
    assert kinds("write to jane.doe+tag@sub.example.test now") == ["email"]
    assert kinds("not an email: @handle or user@ or a@b") == []


def test_credit_card_requires_a_valid_luhn_sum() -> None:
    assert kinds("card 4111 1111 1111 1111 on file") == ["credit_card"]
    assert kinds("card 4111-1111-1111-1111") == ["credit_card"]
    assert "credit_card" not in kinds("order 4111 1111 1111 1112")


def test_iban_detection_and_checksum() -> None:
    assert kinds("pay to GB82 WEST 1234 5698 7654 32 today") == ["iban"]
    assert "iban" not in kinds("pay to GB82 WEST 1234 5698 7654 33")


def test_ip_detection() -> None:
    assert kinds("from 203.0.113.42 and 10.0.0.1") == ["ip", "ip"]
    assert kinds("v6 2001:db8::1 here") == ["ip"]
    assert kinds("version 1.2.3 and 999.1.1.1 and time 12:30:45") == []


def test_phone_detection() -> None:
    assert kinds("call +1 415 555 0132 please") == ["phone"]
    assert kinds("call (020) 7946 0958") == ["phone"]
    assert kinds("call 415-555-0132") == ["phone"]
    assert kinds("intl +442079460958") == ["phone"]
    assert kinds("on 2024-01-15 order 12345 had 3 items") == []


@pytest.mark.parametrize(
    "key",
    [
        FAKE_KEY,
        "sk-abcdefghijklmnop1234",
        "sk-proj-abcdefghijklmnop1234",
        "ghp_" + "a" * 36,
        "AKIA" + "A" * 16,
        "xoxb-1234567890-abcdef",
    ],
)
def test_api_key_detection(key: str) -> None:
    assert kinds(f"token={key}") == ["api_key"]


def test_overlaps_prefer_the_higher_priority_kind() -> None:
    # the digits of the key and the address must not also show up as a phone / email
    text = f"key {FAKE_KEY} mail a@example.test"
    assert kinds(text) == ["api_key", "email"]


def test_kinds_filter_and_validation() -> None:
    text = "a@example.test and 203.0.113.42"
    assert [m.kind for m in find_pii(text, ["ip"])] == ["ip"]
    with pytest.raises(ValueError, match="unknown PII kind"):
        find_pii(text, ["ssn"])
    assert set(PII_KINDS) == {"api_key", "email", "iban", "credit_card", "ip", "phone"}


SAMPLE = (
    "Refund Jordan: jordan@example.test, +1 415 555 0132. Card 4111 1111 1111 1111, "
    f"IBAN GB82 WEST 1234 5698 7654 32, from 203.0.113.42, key {FAKE_KEY}. Again jordan@example.test."
)


def test_redact_restore_round_trip() -> None:
    r = Redactor()
    out = r.redact(SAMPLE)
    for original in ("jordan@example.test", "4111", "GB82", "203.0.113.42", "nvapi-", "415 555"):
        assert original not in out.text
    assert out.counts == {
        "email": 2,
        "phone": 1,
        "credit_card": 1,
        "iban": 1,
        "ip": 1,
        "api_key": 1,
    }
    assert r.restore(out.text) == SAMPLE
    assert restore(out.text, out.mapping) == SAMPLE


def test_same_value_gets_the_same_token_and_mapping_persists() -> None:
    r = Redactor()
    out = r.redact(SAMPLE)
    assert out.text.count("[EMAIL_1]") == 2 and "[EMAIL_2]" not in out.text
    second = r.redact("ping other@example.test and jordan@example.test")
    assert second.text == "ping [EMAIL_2] and [EMAIL_1]"
    assert r.mapping["[EMAIL_2]"] == "other@example.test"
    assert r.restore("[EMAIL_1] / [EMAIL_2]") == "jordan@example.test / other@example.test"
    r.clear()
    assert r.mapping == {} and r.redact("x@example.test").text == "[EMAIL_1]"


def test_label_and_mask_styles_are_one_way() -> None:
    label = redact("mail a@example.test", style="label")
    assert label.text == "mail [EMAIL]" and label.mapping == {}
    mask = redact("mail a@example.test", style="mask")
    assert set(mask.text[5:]) == {"█"} and len(mask.text) == len("mail a@example.test")
    with pytest.raises(ValueError, match="style"):
        Redactor(style="rot13")  # type: ignore[arg-type]


def test_restore_ignores_unknown_tokens_and_empty_mappings() -> None:
    assert restore("see [EMAIL_9]", {"[EMAIL_1]": "a@b.example"}) == "see [EMAIL_9]"
    assert restore("plain", {}) == "plain"
    assert Redactor().restore("[PHONE_1]") == "[PHONE_1]"


def test_text_without_pii_is_unchanged() -> None:
    out = redact("Nothing to see here, just 42 apples.")
    assert (
        out.text == "Nothing to see here, just 42 apples."
        and out.matches == []
        and out.counts == {}
    )


def test_match_offsets_point_at_the_original_text() -> None:
    out = redact(SAMPLE)
    for m in out.matches:
        assert SAMPLE[m.start : m.end] == m.text
