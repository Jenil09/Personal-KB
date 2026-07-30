"""Workspace-wide pytest configuration.

Points testcontainers at rootless Podman. Without this, integration tests either
fail to find a container runtime or hang on the Ryuk reaper sidecar, which is
unreliable rootless. See ai-kb/DECISIONS.md AD-015.
"""

import os
from pathlib import Path


def _podman_socket() -> str | None:
    socket = Path(f"/run/user/{os.getuid()}/podman/podman.sock")
    return f"unix://{socket}" if socket.exists() else None


def pytest_configure() -> None:
    # Respect an explicit DOCKER_HOST if the developer set one.
    if "DOCKER_HOST" not in os.environ and (socket := _podman_socket()):
        os.environ["DOCKER_HOST"] = socket

    os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
