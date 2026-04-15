from __future__ import annotations

from functools import lru_cache

import numpy as np

from .runtime_compat import prepare_openmp_runtime


EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def _load_embedding_model():
    prepare_openmp_runtime()
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is required for baseline ingestion embeddings."
        ) from exc
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _load_embedding_model()
    print(f"[baseline_rag] Embedding {len(texts)} text block(s)...")
    vectors = model.encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")
