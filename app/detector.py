# ===========================================
# Prompt Injection Detector — Main Detector
# ===========================================

"""
Main detection orchestrator.

Combines heuristic rule analysis with optional ML classification
to produce a final risk score and detailed analysis report.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from app.ml_model import MLClassifier
from app.rules import RuleMatch, analyze_with_rules, compute_heuristic_score
from app.schemas import AnalyzeResponse, RuleMatchResponse, StatsResponse

logger = logging.getLogger(__name__)

# Score thresholds for risk classification
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.85,
}

# Weights for combining scores
HEURISTIC_WEIGHT = 0.6
ML_WEIGHT = 0.4


def classify_risk(score: float) -> str:
    """
    Classify a risk score into a human-readable level.

    Args:
        score: Risk score between 0.0 and 1.0.

    Returns:
        One of: "low", "medium", "high", "critical".
    """
    if score < RISK_THRESHOLDS["low"]:
        return "low"
    elif score < RISK_THRESHOLDS["medium"]:
        return "medium"
    elif score < RISK_THRESHOLDS["high"]:
        return "high"
    else:
        return "critical"


class AnalysisStats:
    """Thread-safe in-memory statistics tracker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_analyzed: int = 0
        self._threats_detected: int = 0
        self._total_score: float = 0.0
        self._category_breakdown: dict[str, int] = {}

    def record(self, response: AnalyzeResponse) -> None:
        """Record analysis results into statistics."""
        with self._lock:
            self._total_analyzed += 1
            self._total_score += response.score

            if response.is_injection:
                self._threats_detected += 1

            for rule in response.matched_rules:
                cat = rule.category
                self._category_breakdown[cat] = self._category_breakdown.get(cat, 0) + 1

    def get_stats(self) -> StatsResponse:
        """Return current statistics snapshot."""
        with self._lock:
            avg_score = (
                round(self._total_score / self._total_analyzed, 4)
                if self._total_analyzed > 0
                else 0.0
            )
            return StatsResponse(
                total_analyzed=self._total_analyzed,
                threats_detected=self._threats_detected,
                avg_score=avg_score,
                category_breakdown=dict(self._category_breakdown),
            )


class Detector:
    """
    Main prompt injection detector.

    Orchestrates heuristic and ML analysis, combines scores,
    and maintains in-memory statistics.
    """

    def __init__(self) -> None:
        self._ml_classifier = MLClassifier()
        self._stats = AnalysisStats()

    @property
    def ml_loaded(self) -> bool:
        """Check if the ML model is loaded."""
        return self._ml_classifier.is_loaded

    def analyze(self, text: str) -> AnalyzeResponse:
        """
        Analyze a single input for prompt injection.

        Args:
            text: The user input to analyze.

        Returns:
            An AnalyzeResponse with score, risk level, and matched rules.
        """
        # Step 1: Heuristic analysis
        rule_matches: list[RuleMatch] = analyze_with_rules(text)
        heuristic_score: float = compute_heuristic_score(rule_matches)

        # Step 2: ML analysis (if model is available)
        ml_score: float | None = self._ml_classifier.predict(text)

        # Step 3: Combine scores
        if ml_score is not None:
            final_score = round(
                (HEURISTIC_WEIGHT * heuristic_score) + (ML_WEIGHT * ml_score),
                4,
            )
        else:
            final_score = heuristic_score

        # Step 4: Classify risk
        risk_level = classify_risk(final_score)
        is_injection = risk_level in ("high", "critical")

        # Step 5: Build response
        matched_rules_response = [
            RuleMatchResponse(
                name=m.name,
                category=m.category,
                severity=m.severity,
                description=m.description,
                matched_text=m.matched_text,
            )
            for m in rule_matches
        ]

        # Truncate displayed input for safety
        display_input = text[:500] + ("..." if len(text) > 500 else "")

        response = AnalyzeResponse(
            input=display_input,
            score=final_score,
            risk_level=risk_level,
            is_injection=is_injection,
            heuristic_score=heuristic_score,
            ml_score=ml_score,
            matched_rules=matched_rules_response,
            num_rules_matched=len(matched_rules_response),
        )

        # Step 6: Record stats
        self._stats.record(response)

        logger.info(
            "Analyzed input (%d chars): score=%.4f risk=%s rules=%d",
            len(text),
            final_score,
            risk_level,
            len(rule_matches),
        )

        return response

    def get_stats(self) -> StatsResponse:
        """Return current analysis statistics."""
        return self._stats.get_stats()
