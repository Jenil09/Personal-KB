"""Fixtures for the client's own suite.

Everything here is the contract and nothing is a consumer's: `FakeService` comes
from `kb_client.testing`, and the settings are the narrow `KbClientSettings`
rather than any tool's. No `isolated_config` equivalent is needed — this model
reads no config file, and the constructor arguments below outrank the
environment.
"""

import pytest

from kb_client.settings import KbClientSettings
from kb_client.testing import API_KEY, FakeService


@pytest.fixture
def service() -> FakeService:
    return FakeService()


@pytest.fixture
def api_key() -> str:
    """The key the fake service accepts."""
    return API_KEY


@pytest.fixture
def settings() -> KbClientSettings:
    return KbClientSettings(base_url="http://kb.test", api_key=API_KEY)
