"""Asking an LLM what a document should be called, typed, and tagged (AD-026).

**This is a suggestion, and the type system says so.** Nothing here returns a
`DocumentMetadata` — the type the ingest path accepts. It returns a
`MetadataSuggestion`, which the metadata screen writes into three editable
fields and the subcommands offer as prompt defaults. The only way a suggestion
reaches `/v1/documents` is through a human pressing the button that reads the
fields back. That is not a convention to be careful about; it is the reason the
two types are different.

**The call goes out from here, not from `kb-api`.** The service runs on a VPS
that is not published to the internet (AD-023) and has no route to a laptop's
LM Studio, so a service-side suggester could never use a local model — which is
the configuration this feature exists to make comfortable. Suggestion also
touches no chunk, no vector, and no audit row: it is an authoring aid for a
human at a terminal, and it produces no request the service sees until ingest.

**One adapter, three targets.** OpenAI, Gemini's compatibility endpoint, and
every local runner speak OpenAI-shaped `/chat/completions`, so this is raw
`httpx` against a configured `base_url` rather than a provider SDK. `kb-cli`
gains no dependency for the feature.

Everything the model says is re-derived against `taxonomy` before it is
believed. A suggester is not an authority: an out-of-list type becomes `note`
and says so, and tags are slugged, de-duplicated, and capped whatever comes
back.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from kb_cli.config import SuggestSettings
from kb_client.client import KbClient
from kb_client.taxonomy import DOCUMENT_TYPES, MAX_TAGS, coerce_type, normalise_tags, normalise_type
from platform_core import ConfigurationError, UpstreamError

__all__ = [
    "MetadataSuggestion",
    "collect_tag_vocabulary",
    "suggest_metadata",
]

_MAX_TITLE_CHARS = 512
"""The same ceiling `ingest.title_for` applies, for the same reason."""

_VOCABULARY_LIMIT = 40

_OUTLINE_LINES = 40

_ELISION = "\n\n… [{omitted:,} characters omitted] …\n\n"

_SCHEMA: dict[str, Any] = {
    # `Any` because this is a JSON Schema document on its way into a request
    # body — its shape is the provider's, not ours.
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "type": {"type": "string", "enum": list(DOCUMENT_TYPES)},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "type", "tags"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = f"""\
You classify documents for a personal technical knowledge base.

Answer with a single JSON object and nothing else. It has exactly three keys:

  "title"  the title the author would have written — a name, not a summary
           sentence, and not a filename. No trailing punctuation.
  "type"   exactly one of: {", ".join(DOCUMENT_TYPES)}
  "tags"   between 2 and {MAX_TAGS} lowercase hyphenated topic tags

Rules for tags: name technologies, systems, and domains that the document is
actually about, not every word it mentions. Prefer a tag that already exists in
the knowledge base over a new synonym for it. No generic tags such as "notes",
"documentation", "misc", or "technical".

If the document is truncated you are told so; classify from what you were given
rather than guessing at what is missing.\
"""


@dataclass(frozen=True, slots=True)
class MetadataSuggestion:
    """What the model proposed, after validation. Never ingested as-is."""

    title: str
    type: str
    tags: tuple[str, ...]
    model: str
    truncated: bool = False
    coerced_type: str | None = None
    """What the model actually said, when it was not in the controlled list.

    Carried so the UI can show the substitution. A suggester that silently
    rewrites its own answer teaches the operator to trust it slightly more than
    it deserves.
    """

    @property
    def notes(self) -> tuple[str, ...]:
        """Short caveats to print or show beside the fields."""
        notes = [f"suggested by {self.model}"]
        if self.truncated:
            notes.append("document truncated for the model")
        if self.coerced_type:
            notes.append(f"type coerced from {self.coerced_type!r}")
        return tuple(notes)


async def collect_tag_vocabulary(
    client: KbClient, *, limit: int = _VOCABULARY_LIMIT
) -> tuple[str, ...]:
    """The tags already in use, most common first.

    Passed into the prompt so "prefer an existing tag" is something the model
    can actually do rather than an instruction it has no way to follow. One
    listing pass over a corpus the PRD sizes at tens of documents, which is why
    this is affordable at all; callers treat a failure as an empty vocabulary
    rather than a failed suggestion.
    """
    counts: dict[str, int] = {}
    async for page in client.iter_documents(page_size=100):
        for document in page.documents:
            for tag in document.tags:
                counts[tag] = counts.get(tag, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(tag for tag, _ in ranked[:limit])


async def suggest_metadata(
    content: str,
    *,
    settings: SuggestSettings,
    known_tags: Sequence[str] = (),
    fallback_title: str = "",
    client: httpx.AsyncClient | None = None,
) -> MetadataSuggestion:
    """Ask the configured model to describe `content`.

    `client` is injected by the tests, which drive this through `MockTransport`
    — the same seam `KbClient` uses. When it is `None` one client is opened for
    the call and closed after it; this is a once-per-invocation operation, so
    the shared-client rule that governs the service's outbound calls does not
    buy anything here.
    """
    if not settings.base_url:
        raise ConfigurationError(
            "Metadata suggestion is not configured. "
            "Set a model endpoint with `kb config set suggest.base_url …` "
            "(http://localhost:1234/v1 for LM Studio, "
            "https://api.openai.com/v1 for OpenAI).",
            context={"setting": "suggest.base_url"},
        )
    if not content.strip():
        raise ConfigurationError(
            "There is nothing to suggest metadata for — the document is empty."
        )

    reduced, truncated = _reduce(content, settings.max_content_chars)
    messages = _messages(reduced, known_tags=known_tags, truncated=truncated)

    if client is not None:
        payload = await _post(client, settings, messages)
    else:
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings.timeout_seconds)) as opened:
            payload = await _post(opened, settings, messages)

    return _interpret(
        payload, settings=settings, truncated=truncated, fallback_title=fallback_title
    )


# ---------------------------------------------------------------------------
# the call
# ---------------------------------------------------------------------------


async def _post(
    client: httpx.AsyncClient,
    settings: SuggestSettings,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    """One completion, retried once in a laxer JSON mode.

    `json_schema` is what makes a small local model reliable, and it is also the
    thing least reliably supported by one: llama.cpp-backed runtimes answer 400
    for a `response_format` they do not implement. Falling back to
    `json_object`, and then to lenient parsing, is three layers each of which is
    cheap — and the third is what makes a 7B model usable at all.
    """
    # `Any` throughout: this is a third-party JSON response being narrowed.
    schema_format: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {"name": "document_metadata", "strict": True, "schema": _SCHEMA},
    }
    try:
        return await _completion(client, settings, messages, schema_format)
    except _ResponseFormatError:
        return await _completion(client, settings, messages, {"type": "json_object"})


class _ResponseFormatError(Exception):
    """The endpoint does not implement the `response_format` that was sent."""


async def _completion(
    client: httpx.AsyncClient,
    settings: SuggestSettings,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
) -> dict[str, Any]:
    # `Any` for the same reason as `_post`.
    headers = {"Content-Type": "application/json"}
    if settings.api_key is not None:
        headers["Authorization"] = f"Bearer {settings.api_key.get_secret_value()}"

    body: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        # Zero, and not a matter of taste: a suggester that returns different
        # tags for the same document on two presses is one you stop trusting,
        # and the tests assert on exact output.
        "temperature": 0,
        "response_format": response_format,
    }

    try:
        response = await client.post(
            f"{settings.base_url}/chat/completions",
            json=body,
            headers=headers,
            timeout=httpx.Timeout(settings.timeout_seconds),
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        # Distinguished from every other failure because the cause is almost
        # always local and specific: the model server is not running. An
        # `UpstreamError` here would read as "the provider is having a bad day".
        raise ConfigurationError(
            f"Could not reach the suggestion model at {settings.base_url}: {exc}. "
            "Is the model server running?",
            context={"suggest_base_url": settings.base_url or ""},
        ) from exc
    except httpx.HTTPError as exc:
        raise UpstreamError(f"The suggestion model did not answer: {exc}") from exc

    if response.status_code == httpx.codes.BAD_REQUEST and "response_format" in response.text:
        raise _ResponseFormatError
    if response.status_code >= httpx.codes.BAD_REQUEST:
        raise UpstreamError(
            f"The suggestion model answered {response.status_code}: {_complaint(response)}",
            context={"model": settings.model, "status": response.status_code},
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise UpstreamError("The suggestion model's response was not JSON.") from exc
    if not isinstance(payload, dict):
        raise UpstreamError("The suggestion model's response was not a JSON object.")
    return payload


def _complaint(response: httpx.Response) -> str:
    """The provider's own error sentence, when it wrote one."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200].strip() or response.reason_phrase
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
    return str(body)[:200]


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------


def _messages(content: str, *, known_tags: Sequence[str], truncated: bool) -> list[dict[str, str]]:
    system = _SYSTEM_PROMPT
    if known_tags:
        system += "\n\nTags already used in this knowledge base:\n" + ", ".join(known_tags)
    user = content if not truncated else f"[This document has been truncated.]\n\n{content}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _reduce(content: str, max_chars: int) -> tuple[str, bool]:
    """`content` cut to size, and whether anything was cut.

    Deterministic — no randomness, no clock, nothing derived from the
    environment — for the same reason the chunker is (AD-008), plus one specific
    to here: the tests assert on exact prompt strings, and a reduction that
    varied would make them assert on nothing.

    The document's headings are kept in full ahead of the excerpt. They are the
    highest signal per character for both title and type, they are what a human
    skims to answer the same question, and keeping them means the middle of a
    long document is not simply invisible.
    """
    if len(content) <= max_chars:
        return content, False

    outline = _outline(content)
    budget = max(max_chars - len(outline), max_chars // 2)
    head_chars = (budget * 3) // 4
    tail_chars = budget - head_chars
    head = content[:head_chars].rstrip()
    tail = content[len(content) - tail_chars :].lstrip()
    excerpt = head + _ELISION.format(omitted=len(content) - head_chars - tail_chars) + tail
    return (f"{outline}\n\n{excerpt}" if outline else excerpt), True


def _outline(content: str) -> str:
    headings = [
        stripped
        for line in content.splitlines()
        if (stripped := line.strip()).startswith("#") and stripped.lstrip("#").startswith(" ")
    ]
    if not headings:
        return ""
    kept = headings[:_OUTLINE_LINES]
    if len(headings) > _OUTLINE_LINES:
        kept.append(f"… and {len(headings) - _OUTLINE_LINES} further headings")
    return "OUTLINE:\n" + "\n".join(kept)


# ---------------------------------------------------------------------------
# the answer
# ---------------------------------------------------------------------------


def _interpret(
    payload: dict[str, Any],
    *,
    settings: SuggestSettings,
    truncated: bool,
    fallback_title: str,
) -> MetadataSuggestion:
    # `Any` for the same reason as `_post`.
    suggested = _decode(payload)

    raw_title = str(suggested.get("title") or "").strip()
    title = " ".join(raw_title.split())[:_MAX_TITLE_CHARS] or fallback_title.strip()
    if not title:
        raise UpstreamError("The suggestion model returned no usable title.")

    raw_type = suggested.get("type")
    raw_type_text = str(raw_type).strip() if raw_type is not None else ""
    canonical = normalise_type(raw_type_text)

    raw_tags = suggested.get("tags")
    tags = normalise_tags(str(tag) for tag in raw_tags) if isinstance(raw_tags, list) else ()

    return MetadataSuggestion(
        title=title,
        type=canonical or coerce_type(None),
        tags=tags,
        model=settings.model,
        truncated=truncated,
        # Only when something was actually substituted — a model that answered
        # `note` correctly must not be reported as having been corrected.
        coerced_type=raw_type_text if canonical is None and raw_type_text else None,
    )


def _decode(payload: dict[str, Any]) -> dict[str, Any]:
    """The JSON object the model wrote, out of the completion envelope."""
    # `Any` for the same reason as `_post`.
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UpstreamError("The suggestion model returned no choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise UpstreamError("The suggestion model returned an empty message.")
    extracted = _extract_object(text)
    if extracted is None:
        raise UpstreamError(f"The suggestion model did not return JSON: {text[:200].strip()}")
    return extracted


def _extract_object(text: str) -> dict[str, Any] | None:
    """The first balanced JSON object in `text`, or `None`.

    Lenient on purpose, and only reachable when the strict modes above were
    refused. A small local model asked for JSON routinely answers with a fenced
    block, or a sentence of preamble, or both — all of which contain a perfectly
    good object that a bare `json.loads` would reject.
    """
    # `Any` for the same reason as `_post`.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    try:
                        decoded = json.loads(text[start : index + 1])
                    except ValueError:
                        break
                    if isinstance(decoded, dict):
                        return decoded
                    break
        start = text.find("{", start + 1)
    return None
