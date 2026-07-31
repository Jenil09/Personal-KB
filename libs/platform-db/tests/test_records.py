"""The tier-1 record, and the one field that must never hold what it describes."""

import hashlib

import pytest
from pydantic import ValidationError

from platform_db import AuditRecord, Outcome, fingerprint_credential


def test_a_rejected_credential_is_never_stored() -> None:
    # Failed auth is logged (AD-013), which means the presented key reaches the
    # audit path. Storing it would turn the trail into a credential dump.
    secret = "sk-live-do-not-store-this"

    fingerprint = fingerprint_credential(secret)

    assert secret not in fingerprint
    assert len(fingerprint) == 8
    assert fingerprint == hashlib.sha256(secret.encode()).hexdigest()[:8]


def test_the_same_bad_key_fingerprints_the_same_way() -> None:
    # The point of keeping anything at all: one credential retried a thousand
    # times is visible as one credential.
    assert fingerprint_credential("guess-1") == fingerprint_credential("guess-1")
    assert fingerprint_credential("guess-1") != fingerprint_credential("guess-2")


def test_an_auth_failure_carries_no_key_id(make_record) -> None:
    record = make_record(key_id=None, outcome=Outcome.AUTH_FAILED, status_code=401)

    assert record.to_row()["key_id"] is None
    assert record.to_row()["outcome"] == "auth_failed"


def test_absent_optional_columns_stay_absent(make_record) -> None:
    row = make_record().to_row()

    assert row["client_ip"] is None
    assert row["payload"] is None
    assert row["repeat_burst"] is False


def test_records_are_frozen(make_record) -> None:
    # A record mutated between the failed insert and the spill write would be
    # two different rows depending on which surface it landed on.
    record = make_record()

    with pytest.raises(ValidationError):
        record.latency_ms = 0


def test_the_record_round_trips_through_json(make_record) -> None:
    original = make_record(client_ip="203.0.113.7", payload={"query": "x"})

    assert AuditRecord.model_validate_json(original.model_dump_json()) == original
