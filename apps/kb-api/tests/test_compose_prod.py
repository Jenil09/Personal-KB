"""AD-023's exposure rule, enforced where a change to it would be noticed.

The decision says the absence of `ports:` is the control, and that the absence is
worth a check "because the file is the only place the property can be broken and
a single line breaks it silently". This is that check. It runs in the ordinary
unit suite, so a `ports:` line added for a debugging session fails CI on the
commit that adds it rather than during an incident six weeks later.

Two readings of the same file, deliberately. The textual one is the property as
AD-023 states it — greppable, and true of the file a reviewer reads. The
structural one is the property as Docker acts on it, which a commented-out or
oddly-indented line could satisfy while the text check did not. Neither subsumes
the other.

The rest of the assertions are the network shape from Design §6, which is what
turns "no host port" into "each consumer reaches kb-api and nothing else": n8n
cannot address Postgres because it is not on `kb-internal`, and nothing on
`kb-internal` can reach the internet because it is `internal: true`.

`deploy/verify-exposure.sh` asserts the same properties against a *running*
stack. A file that says the right thing and a daemon that does the right thing
are different claims.
"""

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import yaml


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "uv.lock").exists():
            return candidate
    raise RuntimeError("workspace root not found")


COMPOSE_PATH = _repo_root() / "compose.prod.yml"


@pytest.fixture(scope="module")
def raw() -> str:
    return COMPOSE_PATH.read_text()


@pytest.fixture(scope="module")
def compose(raw: str) -> dict[str, Any]:
    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict)
    return parsed


def test_the_file_contains_no_ports_key(raw: str) -> None:
    """The grep AD-023 asks for, as a test rather than as a habit."""
    offenders = [
        f"{number}: {line.rstrip()}"
        for number, line in enumerate(raw.splitlines(), start=1)
        if line.lstrip().startswith("ports:")
    ]
    assert not offenders, (
        "compose.prod.yml declares a host port binding, which AD-023 forbids "
        "outright — Docker's DNAT rules sit ahead of the host firewall, so this "
        "publishes the service to the internet whatever `ufw` says:\n  " + "\n  ".join(offenders)
    )


def test_no_service_publishes_a_port(compose: dict[str, Any]) -> None:
    """The same property as Docker would parse it, not as a reviewer reads it."""
    published = {
        name: service["ports"]
        for name, service in compose["services"].items()
        if service.get("ports")
    }
    assert not published, f"services publishing host ports: {published}"


def test_the_datastores_are_on_an_internal_network_only(compose: dict[str, Any]) -> None:
    """Postgres and Chroma reachable by kb-api and by nothing else.

    `internal: true` is what makes this stronger than "no published port": it
    also means a compromised datastore container has no route out to the
    internet, which no absence of a `ports:` key would give.
    """
    assert compose["networks"]["kb-internal"]["internal"] is True

    for name in ("postgres", "chroma"):
        assert compose["services"][name]["networks"] == ["kb-internal"], (
            f"{name} must sit on kb-internal alone; another network is a route "
            "off the host or a path from n8n"
        )


def test_kb_api_sits_on_every_network(compose: dict[str, Any]) -> None:
    """The asymmetry is the design (Design §6): kb-api on all four, everything
    else on exactly one. Each consumer and each dependency reaches kb-api and
    nothing else."""
    assert set(compose["services"]["kb-api"]["networks"]) == {
        "kb-internal",
        "kb-shared",
        "kb-tailnet",
        "egress",
    }


def test_the_proxy_target_is_unambiguous_on_the_tailnet(compose: dict[str, Any]) -> None:
    """The tailscale container's `hostname` is the tailnet node name — and Docker
    registers a container's hostname as a resolvable name on its networks. With
    the default `kb-api`, the bare name is ambiguous on kb-tailnet and the
    resolver may answer with the proxy's own address, at which point `serve`
    connects to itself and every operator request is a 502 with the whole stack
    healthy. The alias is the disambiguator; serve.json must use it."""
    serve = json.loads((_repo_root() / "deploy" / "tailscale-serve.json").read_text())
    handler = serve["Web"]["${TS_CERT_DOMAIN}:443"]["Handlers"]["/"]
    target = urlsplit(handler["Proxy"]).hostname

    kb_api = compose["services"]["kb-api"]["networks"]
    aliases = kb_api["kb-tailnet"]["aliases"]

    assert target in aliases, (
        f"serve.json proxies to {target!r}, which is not an alias of kb-api on "
        "kb-tailnet"
    )
    tailnet_hostname = compose["services"]["tailscale"]["hostname"]
    assert target not in tailnet_hostname, (
        f"the proxy target {target!r} collides with the tailscale container's "
        f"hostname {tailnet_hostname!r}; the proxy will resolve to itself"
    )


def test_the_shared_network_is_external(compose: dict[str, Any]) -> None:
    """n8n owns `kb-shared`. This stack joins it; creating it here would mean a
    `compose down` taking n8n's network with it."""
    assert compose["networks"]["kb-shared"]["external"] is True


def test_the_tailscale_container_needs_no_kernel_privileges(compose: dict[str, Any]) -> None:
    """AD-023 chose userspace networking so the proxy needs neither NET_ADMIN nor
    /dev/net/tun. A container holding NET_ADMIN on this host is a larger grant
    than the thing it exists to do."""
    tailscale = compose["services"]["tailscale"]

    assert tailscale["environment"]["TS_USERSPACE"] == "true"
    assert "cap_add" not in tailscale
    assert not any("/dev/net/tun" in str(volume) for volume in tailscale.get("volumes", []))
    # Sharing kb-api's namespace would replace its networks wholesale and take
    # the n8n path down with it — the alternative AD-023 rejected by name.
    assert "network_mode" not in tailscale


def test_migrations_run_before_the_service_starts(compose: dict[str, Any]) -> None:
    """An init container, not app startup: a migration run from the lifespan
    handler turns a failing migration into a crash loop that retries it every few
    seconds."""
    assert (
        compose["services"]["kb-api"]["depends_on"]["migrate"]["condition"]
        == "service_completed_successfully"
    )


def test_the_audit_spill_survives_the_container(compose: dict[str, Any]) -> None:
    """AD-013's guarantee is durability. An anonymous volume here would mean a
    stack replacement during a Postgres outage loses exactly the records the
    spill exists to keep."""
    mounts = compose["services"]["kb-api"]["volumes"]
    assert any(str(mount).startswith("kb-audit-spill:") for mount in mounts)
    assert "kb-audit-spill" in compose["volumes"]


def test_every_long_running_service_restarts(compose: dict[str, Any]) -> None:
    """AD-022 dropped the systemd units; `restart: unless-stopped` under the
    host's Docker service is what replaced them. `migrate` is the exception and
    has to be — restarting a completed init container reruns the migration."""
    for name, service in compose["services"].items():
        expected = "no" if name == "migrate" else "unless-stopped"
        assert service["restart"] == expected, f"{name} has restart: {service['restart']}"
