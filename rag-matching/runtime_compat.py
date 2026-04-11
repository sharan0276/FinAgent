from __future__ import annotations

import os
import sys


def prepare_openmp_runtime() -> None:
    """
    Windows compatibility shim for FAISS / PyTorch OpenMP conflicts.

    On some Windows setups, importing FAISS and Torch-backed libraries in the
    same process can load multiple OpenMP runtimes (`libiomp5md.dll` and
    `libomp140.x86_64.dll`). We keep the FAISS-based path, but set the widely
    used compatibility flag before those imports happen.
    """
    if sys.platform.startswith("win"):
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
