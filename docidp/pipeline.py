"""route -> extract -> verify -> (repair) orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .config import Settings
from .llm import LLMBackend
from .parsers import Document
from .schemas import RouteDecision
from .validators import Issue, format_feedback, normalize_records, validate_records


@dataclass
class AttemptTrace:
    attempt: int
    records: List[Dict[str, Any]]
    issues: List[Issue]

    @property
    def error_count(self) -> int:
        return sum(i.severity == "error" for i in self.issues)


@dataclass
class ExtractionResult:
    source: str
    route: RouteDecision
    records: List[Dict[str, Any]]
    issues: List[Issue]
    attempts: int
    needs_review: bool
    review_reasons: List[str]
    truncated: bool = False
    timings: Dict[str, float] = field(default_factory=dict)
    history: List[AttemptTrace] = field(default_factory=list)

    @property
    def document_type(self) -> str:
        return self.route.document_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "document_type": self.document_type,
            "router_confidence": self.route.confidence,
            "router_reasoning": self.route.reasoning,
            "records": self.records,
            "issues": [i.to_dict() for i in self.issues],
            "attempts": self.attempts,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
            "truncated": self.truncated,
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
        }


class DocumentPipeline:
    def __init__(self, backend: LLMBackend, settings: Settings):
        self.backend = backend
        self.settings = settings

    def run(self, doc: Document, instruction: str = "") -> ExtractionResult:
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        route = self.backend.route(doc)
        route_s = time.perf_counter() - t0
        doc_type = route.document_type

        history: List[AttemptTrace] = []
        feedback = None
        extract_s = 0.0
        for attempt in range(1, self.settings.max_repairs + 2):
            t0 = time.perf_counter()
            raw = self.backend.extract(doc, doc_type, instruction, feedback)
            extract_s += time.perf_counter() - t0

            records, norm_issues = normalize_records(doc_type, raw.model_dump()["items"])
            issues = norm_issues + validate_records(doc_type, records)
            trace = AttemptTrace(attempt, records, issues)
            history.append(trace)
            if trace.error_count == 0:
                break
            feedback = format_feedback(issues)

        # Keep the attempt with the fewest errors; ties keep the earliest (a repair must earn its place).
        best = min(history, key=lambda t: (t.error_count, t.attempt))

        reasons: List[str] = []
        if best.error_count:
            reasons.append(f"{best.error_count} validation error(s) remain after {len(history)} attempt(s)")
        if route.confidence < self.settings.min_router_confidence:
            reasons.append(f"low routing confidence ({route.confidence:.2f})")
        if doc.truncated:
            reasons.append("document text was truncated")

        return ExtractionResult(
            source=doc.name,
            route=route,
            records=best.records,
            issues=best.issues,
            attempts=len(history),
            needs_review=bool(reasons),
            review_reasons=reasons,
            truncated=doc.truncated,
            timings={
                "parse_s": doc.parse_seconds,
                "route_s": route_s,
                "extract_s": extract_s,
                "total_s": doc.parse_seconds + (time.perf_counter() - t_start),
            },
            history=history,
        )
