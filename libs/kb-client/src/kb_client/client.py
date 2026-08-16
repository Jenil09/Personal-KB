"""The `/v1` client — one `httpx.AsyncClient`, problem+json mapped back to errors.

Async throughout, because every consumer already is: `kb-cli`'s Textual app
awaits it on its own loop and its subcommands wrap it in `asyncio.run`. Holding
a second, synchronous client for the synchronous callers would mean two code
paths to the same endpoints.

**Errors arrive as `PlatformError` subclasses, not as status codes.** The service
raises `NotFoundError`, serialises it to RFC 9457, and this reverses that: the
`code` extension member — stable contract, per `platform_core.errors` — picks the
class back out. The result is that a caller here writes `except NotFoundError`
and means the same thing the service meant, and the `detail` the operator reads
is the sentence the service wrote rather than one this file invented.

One client per process, opened in a context manager. Explicit timeouts on every
call, and ingest gets a longer one than the rest (see `KbClientSettings`).
"""

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Self
from uuid import UUID

import httpx

from kb_client.models import DocumentDetail, DocumentPage, IngestResult, SearchResponse, Stats
from kb_client.settings import KbClientSettings
from platform_core import (
    AuthenticationError,
    AuthorizationError,
    ConflictError,
    NotFoundError,
    PayloadTooLargeError,
    PlatformError,
    RateLimitedError,
    UpstreamError,
    ValidationError,
    new_request_id,
)

__all__ = ["KbClient", "open_client"]

_BY_CODE: dict[str, type[PlatformError]] = {
    cls.code: cls
    for cls in (
        AuthenticationError,
        AuthorizationError,
        ConflictError,
        NotFoundError,
        PayloadTooLargeError,
        RateLimitedError,
        UpstreamError,
        ValidationError,
    )
}

_BY_STATUS: dict[int, type[PlatformError]] = {cls.status_code: cls for cls in _BY_CODE.values()}


class KbClient:
    """Everything a consumer asks of `kb-api`, and nothing else."""

    def __init__(
        self, settings: KbClientSettings, transport: httpx.AsyncBaseTransport | None = None
    ):
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=f"{settings.base_url}/v1",
            # AD-011's scheme. The key is attached once, here, rather than by
            # each call site — a request that forgot it would get a 401 that
            # looks exactly like a wrong key.
            headers={"Authorization": f"Bearer {settings.api_key.get_secret_value()}"},
            timeout=httpx.Timeout(settings.timeout_seconds),
            # Injected by the tests, which serve the `/v1` contract — including
            # real problem+json bodies — over `MockTransport`. Deliberately not
            # the `kb-api` app: this package does not depend on the service, and
            # a test that imported it would be asserting the client agrees with
            # one build of the server rather than with the contract.
            transport=transport,
            follow_redirects=False,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- documents --------------------------------------------------------

    async def list_documents(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        document_type: str | None = None,
        source: str | None = None,
        tags: Sequence[str] = (),
        match_all_tags: bool = False,
        collection: str | None = None,
    ) -> DocumentPage:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        # `type` is the wire name (PRD §6.4); `document_type` is the Python one,
        # because the builtin must not be shadowed.
        if document_type:
            params["type"] = document_type
        if source:
            params["source"] = source
        if collection:
            params["collection"] = collection
        if tags:
            params["tags"] = list(tags)
            params["match_all_tags"] = match_all_tags
        return DocumentPage.model_validate(await self._request("GET", "/documents", params=params))

    async def iter_documents(
        self, *, page_size: int = 100, **filters: Any
    ) -> AsyncIterator[DocumentPage]:
        """Every page matching `filters`, oldest request first.

        `total` rather than a short page decides when to stop: the last page
        being exactly full is normal, and treating a full page as "not the end"
        is the bug that makes a 100-document corpus list 100 documents forever.
        """
        # `Any` because these are `list_documents`' own keyword arguments,
        # forwarded unchanged.
        offset = 0
        while True:
            page = await self.list_documents(limit=page_size, offset=offset, **filters)
            yield page
            if not page.has_more or not page.documents:
                return
            offset += len(page.documents)

    async def get_document(self, document_id: UUID) -> DocumentDetail:
        return DocumentDetail.model_validate(
            await self._request("GET", f"/documents/{document_id}")
        )

    async def delete_document(self, document_id: UUID) -> bool:
        """`True` if a document was removed, `False` if there was nothing there.

        The endpoint answers `204` either way — it is idempotent by design
        (PRD §6.6) — and reports which happened in `X-Deleted`. Worth returning,
        because "deleted" and "there was nothing to delete" should not read the
        same in a confirmation the operator is about to trust.
        """
        response = await self._send("DELETE", f"/documents/{document_id}")
        # `str()` because httpx types `Headers.get` loosely; the comparison must be
        # a `bool`, not an `Any` that happens to look like one.
        return str(response.headers.get("X-Deleted", "")) == "true"

    async def ingest(
        self,
        *,
        title: str,
        content: str,
        document_type: str,
        source: str | None = None,
        tags: Sequence[str] = (),
        provider: str | None = None,
    ) -> IngestResult:
        body: dict[str, Any] = {
            "title": title,
            "content": content,
            "type": document_type,
            "tags": list(tags),
        }
        if source is not None:
            body["source"] = source
        # Omitted rather than sent as null when unset, so the service applies its
        # own default provider (AD-006) instead of this client asserting one.
        provider = provider or self._settings.provider
        if provider:
            body["provider"] = provider
        payload = await self._request(
            "POST", "/documents", json=body, timeout=self._settings.ingest_timeout_seconds
        )
        return IngestResult.model_validate(payload)

    # --- search and stats -------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_type: str | None = None,
        source: str | None = None,
        tags: Sequence[str] = (),
        match_all_tags: bool = False,
        provider: str | None = None,
    ) -> SearchResponse:
        filters: dict[str, Any] = {}
        if document_type:
            filters["type"] = document_type
        if source:
            filters["source"] = source
        if tags:
            filters["tags"] = list(tags)
            filters["match_all_tags"] = match_all_tags
        body: dict[str, Any] = {"query": query, "top_k": top_k, "filters": filters}
        provider = provider or self._settings.provider
        if provider:
            body["provider"] = provider
        return SearchResponse.model_validate(await self._request("POST", "/search", json=body))

    async def stats(self) -> Stats:
        return Stats.model_validate(await self._request("GET", "/admin/stats"))

    async def health(self) -> dict[str, Any]:
        """`/health`, which sits outside `/v1` and needs no key."""
        # `Any` because the health body is diagnostic detail this tool renders
        # rather than branches on.
        response = await self._client.get(f"{self._settings.base_url}/health")
        # 503 is a legitimate answer here — `unhealthy` is a report, not a
        # transport failure — so the body is read at either status.
        return dict(response.json())

    # --- transport --------------------------------------------------------

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        # `Any` in and out: this is the untyped seam between HTTP and the
        # models, and every caller validates what comes back.
        response = await self._send(method, path, **kwargs)
        return response.json()

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        # `Any` for the same reason as `_request`.
        timeout = kwargs.pop("timeout", None)
        if timeout is not None:
            kwargs["timeout"] = httpx.Timeout(timeout)
        # A fresh id per request, sent rather than left to the service to mint,
        # so a failure the operator reports can be found in the audit trail
        # without correlating on timestamps. It must be a UUID — the service
        # replaces anything else rather than propagating it.
        kwargs.setdefault("headers", {})["X-Request-ID"] = str(new_request_id())
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                f"{self._settings.base_url} did not answer in time.",
                context={"method": method, "path": path},
            ) from exc
        except httpx.HTTPError as exc:
            # The unreachable-service case, which is the common one on a laptop:
            # a stale tailnet session, or `just up` never run.
            raise UpstreamError(
                f"Could not reach {self._settings.base_url}: {exc}",
                context={"method": method, "path": path},
            ) from exc
        if response.is_success:
            return response
        raise _as_error(response)


def _as_error(response: httpx.Response) -> PlatformError:
    """Reverse the problem+json mapping, falling back on the status.

    A response that is not problem+json at all — an HTML error page from
    something in front of the service, most likely — still has to produce a
    usable error, so the status picks the class and the body's first line
    becomes the detail.
    """
    detail = response.text.strip()[:500] or f"HTTP {response.status_code}"
    context: Mapping[str, Any] = {}
    code: str | None = None
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        code = body.get("code")
        detail = str(body.get("detail") or detail)
        context = {
            key: value
            for key, value in body.items()
            if key not in {"type", "title", "status", "detail", "instance", "code"}
        }
    cls = _BY_CODE.get(code or "") or _BY_STATUS.get(response.status_code, PlatformError)
    return cls(detail, context=dict(context))


@asynccontextmanager
async def open_client(
    settings: KbClientSettings, transport: httpx.AsyncBaseTransport | None = None
) -> AsyncIterator[KbClient]:
    client = KbClient(settings, transport=transport)
    try:
        yield client
    finally:
        await client.aclose()
