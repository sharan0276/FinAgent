from __future__ import annotations

import os


def prepare_openmp_runtime() -> None:
    """Reduce Windows OpenMP conflicts when loading FAISS/Torch stacks."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
