"""Bearer authentication and caller attribution (AD-011), through the app.

The registry and the scope rules are `platform_core.auth`'s and are tested
there. What is asserted here is the HTTP behaviour built on them: which status
code a caller sees, what the response says, and which routes the dependency
reaches without being asked for.
"""

import httpx
import pytest
from fastapi import APIRouter, Depends

from platform_fastapi import CurrentPrincipal, require_scope


@pytest.fixture
def protected() -> APIRouter:
    router = APIRouter()

    @router.get("/whoami")
    async def whoami(principal: CurrentPrincipal) -> dict[str, str]:
        return {"key_id": principal.key_id}

    return router


@pytest.fixture
async def client(make_app, client_for, protected):
    async with client_for(make_app(routers=[protected])) as http_client:
        yield http_client


def _auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


async def test_a_valid_key_resolves_to_its_key_id(
    client: httpx.AsyncClient, valid_key: str
) -> None:
    response = await client.get("/v1/whoami", headers=_auth(valid_key))

    assert response.status_code == 200
    assert response.json() == {"key_id": "n8n"}


async def test_each_key_attributes_to_its_own_caller(
    client: httpx.AsyncClient, other_key: str
) -> None:
    response = await client.get("/v1/whoami", headers=_auth(other_key))
    assert response.json() == {"key_id": "cli"}


async def test_health_needs_no_key(client: httpx.AsyncClient) -> None:
    assert (await client.get("/health")).status_code == 200


@pytest.mark.parametrize(
    ("header", "expected_detail"),
    [
        (None, "Authorization header is missing."),
        ("Basic {key}", "Authorization header must be 'Bearer <api key>'."),
        ("{key}", "Authorization header must be 'Bearer <api key>'."),
        ("Bearer", "Authorization header must be 'Bearer <api key>'."),
        ("Bearer ", "Authorization header must be 'Bearer <api key>'."),
        ("Bearer wrong-key", "The presented API key is not recognised."),
    ],
)
async def test_rejected_requests_answer_401(
    client: httpx.AsyncClient, valid_key: str, header: str | None, expected_detail: str
) -> None:
    headers = {} if header is None else {"Authorization": header.format(key=valid_key)}
    response = await client.get("/v1/whoami", headers=headers)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    body = response.json()
    assert body["code"] == "unauthenticated"
    assert body["title"] == "Unauthorized"
    assert body["detail"] == expected_detail


async def test_a_rejection_never_echoes_the_presented_key(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/whoami", headers=_auth("almost-the-right-key"))
    assert "almost-the-right-key" not in response.text


async def test_every_v1_router_is_protected_without_opting_in(
    make_app, client_for, valid_key: str
) -> None:
    """A router declaring no dependency of its own is still behind auth."""
    router = APIRouter()

    @router.get("/unaware")
    async def unaware() -> dict[str, bool]:
        return {"reached": True}

    async with client_for(make_app(routers=[router])) as client:
        assert (await client.get("/v1/unaware")).status_code == 401
        assert (await client.get("/v1/unaware", headers=_auth(valid_key))).status_code == 200


async def test_openapi_documents_the_scheme_and_leaves_health_open(
    client: httpx.AsyncClient,
) -> None:
    schema = (await client.get("/openapi.json")).json()

    assert schema["components"]["securitySchemes"]["API key"] == {
        "type": "http",
        "scheme": "bearer",
        "description": (
            "`Authorization: Bearer <api key>`. Required on every route except `/health`."
        ),
    }
    assert schema["paths"]["/v1/whoami"]["get"]["security"] == [{"API key": []}]
    assert "security" not in schema["paths"]["/health"]["get"]


# --- scopes (AD-024) ------------------------------------------------------


@pytest.fixture
def scoped(make_app, client_for):
    """An app whose one route needs `write`, with `n8n` holding only `search`."""
    router = APIRouter()

    @router.delete("/thing", dependencies=[Depends(require_scope("write"))])
    async def remove() -> dict[str, bool]:
        return {"removed": True}

    @router.get("/thing", dependencies=[Depends(require_scope("search"))])
    async def read() -> dict[str, bool]:
        return {"read": True}

    return client_for(
        make_app(
            routers=[router],
            api_key_scopes={"n8n": frozenset({"search"}), "cli": frozenset({"search", "write"})},
        )
    )


async def test_a_search_scoped_key_cannot_delete(scoped, valid_key: str) -> None:
    """The n8n key is the reason AD-024 exists: it must not be able to purge the corpus."""
    async with scoped as client:
        response = await client.delete("/v1/thing", headers=_auth(valid_key))

    assert response.status_code == 403
    body = response.json()
    assert body["code"] == "insufficient_scope"
    assert body["required_scope"] == "write"


async def test_the_same_key_still_reads(scoped, valid_key: str) -> None:
    async with scoped as client:
        assert (await client.get("/v1/thing", headers=_auth(valid_key))).status_code == 200


async def test_a_write_scoped_key_deletes(scoped, other_key: str) -> None:
    async with scoped as client:
        assert (await client.delete("/v1/thing", headers=_auth(other_key))).status_code == 200


async def test_a_missing_key_is_401_not_403(scoped) -> None:
    """Order matters: authentication runs first, so an anonymous caller learns
    nothing about which scopes a route needs."""
    async with scoped as client:
        response = await client.delete("/v1/thing")

    assert response.status_code == 401
    assert response.json()["code"] == "unauthenticated"


async def test_a_key_with_no_scope_entry_is_unrestricted(make_app, client_for, valid_key) -> None:
    """The permissive default. Configuring keys and forgetting scopes leaves a
    working service rather than one that refuses every valid key."""
    router = APIRouter()

    @router.delete("/thing", dependencies=[Depends(require_scope("write"))])
    async def remove() -> dict[str, bool]:
        return {"removed": True}

    async with client_for(make_app(routers=[router])) as client:
        assert (await client.delete("/v1/thing", headers=_auth(valid_key))).status_code == 200
