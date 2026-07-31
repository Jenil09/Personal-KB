"""Provider drivers. Each imports an SDK that ships as an extra (AD-010).

Nothing is re-exported here on purpose: importing this package must not drag in
an SDK the service did not install. Import the driver directly —
`from ai_embeddings.adapters.openai import OpenAIEmbeddingProvider`.
"""
