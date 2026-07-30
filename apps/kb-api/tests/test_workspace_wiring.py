"""Guards the workspace wiring itself.

If `platform-core` ever resolves from PyPI instead of the local workspace member,
this fails loudly here rather than confusingly somewhere downstream.
"""

from pathlib import Path

import kb_api
import platform_core


def _repo_root() -> Path:
    """Walk up to the workspace root rather than counting parents."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "uv.lock").exists():
            return candidate
    raise RuntimeError("workspace root not found")


REPO_ROOT = _repo_root()


def test_kb_api_imports() -> None:
    assert kb_api.__version__


def test_platform_core_resolves_from_workspace() -> None:
    resolved = Path(platform_core.__file__).resolve()
    expected = REPO_ROOT / "libs" / "platform-core" / "src" / "platform_core" / "__init__.py"
    assert resolved == expected, f"platform_core resolved to {resolved}, not the workspace member"
