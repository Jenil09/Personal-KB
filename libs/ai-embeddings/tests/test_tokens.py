"""Token counting and the pre-flight batching it feeds.

Batching is what keeps an ingest inside the provider's per-request limits
(Design §3.1 step 5). Getting it wrong costs a rejected request at best and, if
the batches lose their order, an index where every vector is attached to the
wrong chunk.
"""

import pytest

from ai_embeddings import batch_texts, count_tokens, encoding_for, total_tokens


def test_counting_uses_the_encoder_the_api_bills_with() -> None:
    # cl100k_base is text-embedding-3-small's tokeniser, which is why our
    # pre-flight numbers and the API's reported usage agree exactly.
    assert encoding_for("cl100k_base").name == "cl100k_base"
    assert count_tokens("hello world") == 2


def test_counting_is_not_a_word_count() -> None:
    # The heuristic this replaces (characters over four) is wrong in both
    # directions, and it is wrong on exactly the technical vocabulary this KB
    # is full of.
    assert count_tokens("supercalifragilisticexpialidocious") > 1
    assert count_tokens("") == 0


def test_totals_add_up() -> None:
    texts = ["alpha beta", "gamma delta epsilon"]

    assert total_tokens(texts) == sum(count_tokens(text) for text in texts)


def test_nothing_to_batch_is_no_batches() -> None:
    assert batch_texts([], max_inputs=96, max_tokens=100_000) == []


def test_a_small_ingest_is_one_request() -> None:
    texts = ["alpha", "beta", "gamma"]

    assert batch_texts(texts, max_inputs=96, max_tokens=100_000) == [texts]


def test_the_input_ceiling_binds() -> None:
    texts = [f"chunk {index}" for index in range(250)]

    batches = batch_texts(texts, max_inputs=96, max_tokens=100_000)

    assert [len(batch) for batch in batches] == [96, 96, 58]


def test_the_token_ceiling_binds_first_when_it_is_tighter() -> None:
    # Long chunks hit the token budget well before the input count, which is
    # why both limits exist rather than just the easier one.
    texts = ["token " * 500] * 20

    batches = batch_texts(texts, max_inputs=96, max_tokens=2_000)

    assert len(batches) > 1
    assert all(total_tokens(batch) <= 2_000 or len(batch) == 1 for batch in batches)


def test_batching_preserves_order_and_loses_nothing() -> None:
    # The caller zips vectors back onto chunks by position; a reordered or
    # short batch corrupts the index silently.
    texts = [f"chunk {index}" for index in range(200)]

    batches = batch_texts(texts, max_inputs=7, max_tokens=100_000)

    assert [text for batch in batches for text in batch] == texts


def test_a_text_over_the_whole_budget_still_gets_a_batch() -> None:
    # Dropping it here would produce a batch one short of the chunk list. The
    # port rejects it instead, with a message naming the input.
    texts = ["alpha", "token " * 5_000, "beta"]

    batches = batch_texts(texts, max_inputs=96, max_tokens=1_000)

    assert [text for batch in batches for text in batch] == texts
    assert any(len(batch) == 1 and batch[0].startswith("token") for batch in batches)


def test_a_zero_input_ceiling_is_rejected() -> None:
    # Would otherwise loop forever producing empty batches.
    with pytest.raises(ValueError, match="max_inputs"):
        batch_texts(["alpha"], max_inputs=0, max_tokens=100)
