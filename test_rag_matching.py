from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import unittest
from pathlib import Path

if sys.platform.startswith("win"):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


RAG_MATCHING_DIR = Path(__file__).resolve().parent / "rag-matching"
if str(RAG_MATCHING_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_MATCHING_DIR))

from curator_store import CURATOR_SOURCE_ROOT  # noqa: E402
from indexer import build_index, load_index  # noqa: E402
from matcher import find_matches  # noqa: E402


class RagMatchingTests(unittest.TestCase):
    def _make_artifact_dir(self, name: str) -> Path:
        base_dir = Path(__file__).resolve().parent / ".tmp_rag_tests"
        artifact_dir = base_dir / name
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(artifact_dir, ignore_errors=True))
        return artifact_dir

    def setUp(self) -> None:
        self.query_file = CURATOR_SOURCE_ROOT / "AAPL" / "aapl_2022.json"
        self.assertTrue(self.query_file.exists(), "Expected AAPL query curator file to exist")

    def test_build_index_discovers_all_curator_files(self) -> None:
        artifact_dir = self._make_artifact_dir("build_index")
        payload = build_index(artifact_dir=artifact_dir)

        self.assertEqual(len(payload["entries"]), 15)
        self.assertTrue((artifact_dir / "metadata.json").exists())
        if payload["backend"] == "faiss":
            self.assertTrue((artifact_dir / "faiss.index").exists())
        else:
            self.assertTrue((artifact_dir / "matrix.npy").exists())

    def test_matcher_returns_two_distinct_non_query_companies(self) -> None:
        artifact_dir = self._make_artifact_dir("match_distinct")
        build_index(artifact_dir=artifact_dir)

        result = find_matches(self.query_file, top_k=2, artifact_dir=artifact_dir)
        tickers = [match["ticker"] for match in result["matches"]]

        self.assertEqual(result["query"]["ticker"], "AAPL")
        self.assertLessEqual(len(tickers), 2)
        self.assertEqual(len(tickers), len(set(tickers)))
        self.assertNotIn("AAPL", tickers)
        self.assertEqual(set(tickers), {"GOOG", "META"})

    def test_matcher_reuses_saved_index(self) -> None:
        artifact_dir = self._make_artifact_dir("reuse_index")
        build_index(artifact_dir=artifact_dir)

        first = find_matches(self.query_file, top_k=2, artifact_dir=artifact_dir)
        second = find_matches(self.query_file, top_k=2, artifact_dir=artifact_dir)

        self.assertEqual(first["matches"], second["matches"])
        bundle = load_index(artifact_dir=artifact_dir)
        self.assertEqual(len(bundle["entries"]), 15)

    def test_query_without_embedding_vector_uses_fallback_when_available(self) -> None:
        if importlib.util.find_spec("sentence_transformers") is None:
            self.skipTest("sentence_transformers is not installed in this environment")

        artifact_dir = self._make_artifact_dir("fallback_query")
        build_index(artifact_dir=artifact_dir)

        query_copy = artifact_dir / "aapl_2022_no_vector.json"
        data = json.loads(self.query_file.read_text(encoding="utf-8"))
        data["embedding_vector"] = []
        query_copy.write_text(json.dumps(data), encoding="utf-8")

        result = find_matches(query_copy, top_k=2, artifact_dir=artifact_dir)
        tickers = [match["ticker"] for match in result["matches"]]

        self.assertNotIn("AAPL", tickers)
        self.assertLessEqual(len(tickers), 2)


if __name__ == "__main__":
    unittest.main()
