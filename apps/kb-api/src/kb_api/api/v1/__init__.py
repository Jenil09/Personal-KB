"""Version 1 of the API (AD-012). `create_app` mounts these under `/v1`."""

from kb_api.api.v1.documents import create_documents_router
from kb_api.api.v1.search import create_search_router

__all__ = ["create_documents_router", "create_search_router"]
