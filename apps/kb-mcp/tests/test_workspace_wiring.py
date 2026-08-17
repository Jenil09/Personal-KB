"""Guards the boundaries this package is defined by.

Two of them, and both are the kind that decay silently. `platform_core` must
resolve from the workspace rather than from an index, the way
`apps/kb-api/tests/test_workspace_wiring.py` checks for the service. And the
import graph must stay clear of the database stack: Stage 2 moved `Principal` and
`ApiKeyRegistry` down into `platform-core` precisely so this image would not
carry SQLAlchemy and asyncpg, and one `from platform_fastapi.auth import ...`
would undo it without breaking a single test.
"""

import sys
import tomllib
from pathlib import Path

import kb_mcp
import platform_core

FORBIDDEN = ("fastapi", "platform_fastapi", "platform_db", "sqlalchemy", "asyncpg", "chromadb")


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "uv.lock").exists():
            return candidate
    raise RuntimeError("workspace root not found")


REPO_ROOT = _repo_root()


def test_kb_mcp_imports() -> None:
    assert kb_mcp.__version__


def test_platform_core_resolves_from_the_workspace() -> None:
    resolved = Path(platform_core.__file__).resolve()
    expected = REPO_ROOT / "libs" / "platform-core" / "src" / "platform_core" / "__init__.py"
    assert resolved == expected, f"platform_core resolved to {resolved}, not the workspace member"


def test_the_server_never_imports_the_database_stack() -> None:
    """Imported in a subprocess, because this suite's other modules import freely.

    `kb_client.testing` and the rest of the workspace are already in
    `sys.modules` by the time a test runs, so asking about this process would
    prove nothing. A fresh interpreter importing only the composition root is the
    question that matters: what does *this app* pull in.
    """
    import subprocess

    probe = (
        "import sys, kb_mcp.main;"
        f"found=[m for m in {FORBIDDEN!r} if m in sys.modules];"
        "print(','.join(found))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "", f"kb_mcp.main imported {result.stdout.strip()}"


def test_kb_api_is_not_a_dependency() -> None:
    """AD-025's boundary, restated for a second consumer.

    `kb-mcp` speaks the published `/v1` contract and must be able to be older than
    the service it talks to. A test importing `kb_api` would assert the two agree
    on this commit, which they trivially do and which proves nothing.
    """
    manifest = (REPO_ROOT / "apps" / "kb-mcp" / "pyproject.toml").read_bytes()
    declared = tomllib.loads(manifest.decode("utf-8"))["project"]["dependencies"]

    named = {requirement.split(">")[0].split("[")[0].strip() for requirement in declared}
    assert named.isdisjoint({"kb-api", "platform-db", "platform-fastapi"})
    assert "kb-client" in named
