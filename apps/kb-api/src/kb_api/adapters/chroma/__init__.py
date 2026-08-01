"""Chroma vector store (AD-010). The client and the store, nothing else."""

from kb_api.adapters.chroma.store import ChromaVectorStore, create_chroma_client

__all__ = ["ChromaVectorStore", "create_chroma_client"]
