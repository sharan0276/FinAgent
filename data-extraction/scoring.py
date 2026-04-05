from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Sequence


class BaseSentenceScorer(ABC):
    model_name: str

    @abstractmethod
    def score_sentences(self, sentences: Sequence[str]) -> List[Dict[str, float]]:
        raise NotImplementedError


class FinBERTSentenceScorer(BaseSentenceScorer):
    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "FinBERT scoring requires 'transformers' and 'torch'. "
                "Install them before running data-extraction."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

    def score_sentences(self, sentences: Sequence[str]) -> List[Dict[str, float]]:
        if not sentences:
            return []
        outputs: List[Dict[str, float]] = []
        for start in range(0, len(sentences), self.batch_size):
            batch = list(sentences[start : start + self.batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256,
            )
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
                probs = self._torch.softmax(logits, dim=-1).tolist()
            for values in probs:
                outputs.append(
                    {
                        "positive": float(values[0]),
                        "negative": float(values[1]),
                        "neutral": float(values[2]),
                    }
                )
        return outputs
