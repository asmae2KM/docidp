"""Offline LLM backends for tests: no network, fully deterministic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from docidp.parsers import Document
from docidp.schemas import RECORD_MODELS, SCHEMA_REGISTRY, RouteDecision


def make_output(doc_type: str, records: List[Dict[str, Any]]):
    """Build a schema-valid extraction object; missing fields become null / []."""
    model = RECORD_MODELS[doc_type]
    full = []
    for rec in records:
        row = {name: ([] if "List" in str(f.annotation) else None) for name, f in model.model_fields.items()}
        row.update(rec)
        full.append(row)
    return SCHEMA_REGISTRY[doc_type].model_validate({"items": full})


class ScriptedBackend:
    """Returns pre-programmed routes/extractions in order and records the feedback it was given."""

    def __init__(self, route: RouteDecision, extractions: List[List[Dict[str, Any]]]):
        self._route = route
        self._extractions = list(extractions)
        self.feedbacks: List[Optional[str]] = []
        self.calls = 0

    def route(self, doc: Document) -> RouteDecision:
        self.calls += 1
        return self._route

    def extract(self, doc, doc_type, instruction="", feedback=None):
        self.calls += 1
        self.feedbacks.append(feedback)
        records = self._extractions.pop(0) if len(self._extractions) > 1 else self._extractions[0]
        return make_output(doc_type, records)


class OracleBackend:
    """Looks the answer up in labels.json by file name. Optionally corrupts the first extraction of chosen ids."""

    def __init__(self, dataset_dir: Path, corrupt_first: Optional[Dict[str, Dict[str, Any]]] = None):
        labels = json.loads((Path(dataset_dir) / "labels.json").read_text(encoding="utf-8"))
        self.by_name = {Path(d["file"]).name: d for d in labels["documents"]}
        self.corrupt_first = corrupt_first or {}
        self.seen: set = set()
        self.calls = 0

    def route(self, doc: Document) -> RouteDecision:
        self.calls += 1
        return RouteDecision(document_type=self.by_name[doc.name]["expected_type"], confidence=0.95, reasoning="oracle")

    def extract(self, doc, doc_type, instruction="", feedback=None):
        self.calls += 1
        gold = [dict(r) for r in self.by_name[doc.name]["records"]] or [{}]
        patch = self.corrupt_first.get(doc.name)
        if patch and doc.name not in self.seen:
            gold[0].update(patch)
        self.seen.add(doc.name)
        return make_output(doc_type, gold)
