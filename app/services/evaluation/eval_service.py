"""
app/services/evaluation/eval_service.py
────────────────────────────────────────
Ragas evaluation pipeline.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.core.plugin_registry import registry
from app.models.domain import EvalMetrics

logger = get_logger(__name__)


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    contexts: list[str]
    answer: str


class EvaluationService:
    async def evaluate_plugin(
        self,
        use_case_id: str,
        samples: list[EvalSample],
    ) -> EvalMetrics:
        plugin = registry.get(use_case_id)
        if plugin is None:
            raise ValueError(f"Unknown use_case_id: {use_case_id}")
        if not samples:
            raise ValueError("No samples provided for evaluation")

        logger.info("eval.start", use_case=use_case_id, sample_count=len(samples))

        try:
            metrics = await self._run_ragas(samples)
        except ImportError:
            logger.warning("eval.ragas_not_available", use_case=use_case_id)
            metrics = await self._mock_eval(samples)

        thresholds = plugin.eval_thresholds
        passed = (
            metrics["faithfulness"] >= thresholds.faithfulness
            and metrics["answer_relevancy"] >= thresholds.answer_relevancy
            and metrics["context_recall"] >= thresholds.context_recall
        )

        result = EvalMetrics(
            use_case_id=use_case_id,
            faithfulness=round(metrics["faithfulness"], 4),
            answer_relevancy=round(metrics["answer_relevancy"], 4),
            context_recall=round(metrics["context_recall"], 4),
            passed=passed,
            sample_size=len(samples),
        )

        logger.info(
            "eval.complete",
            use_case=use_case_id,
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            context_recall=result.context_recall,
            passed=result.passed,
        )
        return result

    async def _run_ragas(self, samples: list[EvalSample]) -> dict[str, float]:
        from datasets import Dataset                          # type: ignore[import]
        from ragas import evaluate                            # type: ignore[import]
        from ragas.metrics import (                          # type: ignore[import]
            answer_relevancy,
            context_recall,
            faithfulness,
        )

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
        }
        dataset = Dataset.from_dict(data)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_recall],
            ),
        )
        return {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_recall": float(result["context_recall"]),
        }

    async def _mock_eval(self, samples: list[EvalSample]) -> dict[str, float]:
        import re

        def overlap_score(answer: str, context: str) -> float:
            answer_words = set(re.findall(r"\w+", answer.lower()))
            context_words = set(re.findall(r"\w+", context.lower()))
            if not answer_words:
                return 0.0
            return len(answer_words & context_words) / len(answer_words)

        faithfulness_scores: list[float] = []
        relevancy_scores: list[float] = []
        recall_scores: list[float] = []

        for s in samples:
            all_context = " ".join(s.contexts)
            faithfulness_scores.append(min(overlap_score(s.answer, all_context) * 1.5, 1.0))
            relevancy_scores.append(min(overlap_score(s.answer, s.question) * 2.0, 1.0))
            recall_scores.append(min(overlap_score(s.ground_truth, all_context) * 1.3, 1.0))

        return {
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
            "answer_relevancy": sum(relevancy_scores) / len(relevancy_scores),
            "context_recall": sum(recall_scores) / len(recall_scores),
        }

    def format_report(self, metrics: EvalMetrics) -> str:
        status = "✓ PASSED" if metrics.passed else "✗ FAILED"
        plugin = registry.get(metrics.use_case_id)
        thresholds = plugin.eval_thresholds if plugin else None

        lines = [
            f"Evaluation Report — {metrics.use_case_id}",
            f"Status: {status}",
            f"Samples: {metrics.sample_size}",
            f"Evaluated: {metrics.evaluated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "Metrics:",
        ]
        for name, val, threshold in [
            ("Faithfulness", metrics.faithfulness, thresholds.faithfulness if thresholds else 0),
            ("Answer Relevancy", metrics.answer_relevancy, thresholds.answer_relevancy if thresholds else 0),
            ("Context Recall", metrics.context_recall, thresholds.context_recall if thresholds else 0),
        ]:
            flag = "✓" if val >= threshold else "✗"
            lines.append(f"  {flag} {name}: {val:.3f} (threshold: {threshold:.2f})")

        return "\n".join(lines)


def get_eval_service() -> EvaluationService:
    return EvaluationService()
