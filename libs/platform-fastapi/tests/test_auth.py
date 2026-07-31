"""Bearer authentication and caller attribution (AD-011)."""

import httpx
import pytest
from fastapi import APIRouter
from pydantic import SecretStr

from platform_core import AuthenticationError
from platform_fastapi import ApiKeyRegistry, CurrentPrincipal, Principal


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


def test_registry_resolves_the_matching_key() -> None:
    registry = ApiKeyRegistry({"n8n": SecretStr("one"), "cli": SecretStr("two")})

    assert registry.resolve("one") == Principal(key_id="n8n")
    assert registry.resolve("two") == Principal(key_id="cli")


def test_registry_rejects_an_unknown_key() -> None:
    registry = ApiKeyRegistry({"n8n": SecretStr("one")})

    with pytest.raises(AuthenticationError) as caught:
        registry.resolve("two")
    assert caught.value.status_code == 401


def test_registry_rejects_a_prefix_of_a_real_key() -> None:
    registry = ApiKeyRegistry({"n8n": SecretStr("secret-value")})

    with pytest.raises(AuthenticationError):
        registry.resolve("secret")
