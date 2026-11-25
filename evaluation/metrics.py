"""Field-level scoring with type-aware matchers. Pure functions, no I/O, fully unit-tested."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from docidp.validators import normalize_date, normalize_datetime

MONEY = {"subtotal", "tax_amount", "total_due", "total_paid"}
FUZZY_TEXT = {"vendor_name": 0.92, "store_name": 0.92, "document_title": 0.85}
EXACT_ID = {"invoice_number"}


def _norm_text(s: Any) -> str:
    return re.sub(r"[\W_]+", " ", str(s).casefold(), flags=re.UNICODE).strip()


def _norm_id(s: Any) -> str:
    return re.sub(r"[\W_]+", "", str(s).casefold(), flags=re.UNICODE)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def list_f1(gold: Sequence[str], pred: Sequence[str], threshold: float = 0.8) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    remaining = [_norm_text(p) for p in pred]
    tp = 0
    for g in map(_norm_text, gold):
        best_i, best = -1, 0.0
        for i, p in enumerate(remaining):
            r = _ratio(g, p)
            if r > best:
                best_i, best = i, r
        if best >= threshold:
            tp += 1
            remaining.pop(best_i)
    precision, recall = tp / len(pred), tp / len(gold)
    return 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)


def score_field(name: str, gold: Any, pred: Any) -> float:
    """1.0 = correct, 0.0 = wrong; list fields return F1 in [0, 1]."""
    if name == "items_bought":
        return list_f1(gold or [], pred or [])
    if name in MONEY:
        if gold is None:  # "no such line on the document": null or 0 are both acceptable
            return 1.0 if pred in (None, 0, 0.0) else 0.0
        p = _to_float(pred)
        return 1.0 if p is not None and abs(p - float(gold)) <= 0.01 else 0.0
    if gold is None:
        return 1.0 if pred in (None, "") else 0.0
    if pred in (None, ""):
        return 0.0
    if name in ("issue_date", "creation_date"):
        return 1.0 if normalize_date(str(pred)) == gold else 0.0
    if name == "transaction_time":
        return 1.0 if normalize_datetime(str(pred)) == gold else 0.0
    if name == "currency":
        return 1.0 if str(pred).strip().upper() == str(gold).upper() else 0.0
    if name in EXACT_ID:
        return 1.0 if _norm_id(pred) == _norm_id(gold) else 0.0
    if name in FUZZY_TEXT:
        g, p = _norm_text(gold), _norm_text(pred)
        return 1.0 if g == p or _ratio(g, p) >= FUZZY_TEXT[name] else 0.0
    return 1.0 if _norm_text(gold) == _norm_text(pred) else 0.0


def _record_key(doc_type: str, rec: Dict[str, Any]) -> str:
    if doc_type == "invoice":
        return _norm_id(rec.get("invoice_number") or "")
    if doc_type == "receipt":
        return _norm_text(f"{rec.get('store_name') or ''} {rec.get('transaction_time') or ''}")
    return _norm_text(rec.get("document_title") or "")


def match_records(
    doc_type: str, gold: List[Dict[str, Any]], pred: List[Dict[str, Any]]
) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """Pair each gold record with at most one predicted record (greedy by key similarity)."""
    pairs = sorted(
        (
            (_ratio(_record_key(doc_type, g), _record_key(doc_type, p)), gi, pi)
            for gi, g in enumerate(gold)
            for pi, p in enumerate(pred)
        ),
        reverse=True,
    )
    min_sim = (
        0.0 if len(gold) == len(pred) == 1 else 0.5
    )  # one-to-one: always pair, so a wrong id is scored as a wrong field
    used_g, used_p, assigned = set(), set(), {}
    for sim, gi, pi in pairs:
        if sim < min_sim or gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        assigned[gi] = pi
    return [(g, pred[assigned[gi]] if gi in assigned else None) for gi, g in enumerate(gold)]


@dataclass
class DocScore:
    fields: List[Tuple[str, float]] = field(default_factory=list)
    n_gold: int = 0
    n_pred: int = 0

    @property
    def exact(self) -> bool:
        return self.n_gold == self.n_pred and all(s >= 1.0 for _, s in self.fields)

    @property
    def accuracy(self) -> float:
        return sum(s for _, s in self.fields) / len(self.fields) if self.fields else 1.0


def score_document(doc_type: str, gold: List[Dict[str, Any]], pred: List[Dict[str, Any]]) -> DocScore:
    out = DocScore(n_gold=len(gold), n_pred=len(pred))
    for g, p in match_records(doc_type, gold, pred):
        for name, gold_val in g.items():
            out.fields.append((name, 0.0 if p is None else score_field(name, gold_val, p.get(name))))
    return out
