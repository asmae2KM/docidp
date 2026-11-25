"""Deterministic post-processing and consistency checks.

This is the "verifier" half of the extract -> verify -> repair loop. Nothing here calls an LLM.
Issues with severity "error" trigger a repair attempt; "warning" issues are surfaced to the user.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    "%m/%d/%Y",  # tried last: only matches when day-first failed (e.g. 03/14/2026)
]  # fmt: skip
TIME_FORMATS = ["%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p", "%I:%M%p"]
DATE_FIELDS = {"issue_date", "creation_date"}
DATETIME_FIELDS = {"transaction_time"}
CURRENCY_SYMBOLS = {"€": "EUR", "£": "GBP", "EURO": "EUR", "EUROS": "EUR", "DH": "MAD", "DHS": "MAD"}


@dataclass(frozen=True)
class Issue:
    severity: str  # "error" | "warning"
    field: str
    message: str
    record_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return f"[record {self.record_index + 1}] {self.field}: {self.message}"


# ------------------------------------------------------------------ normalisation
def normalize_date(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    v = re.sub(r"\s+", " ", value.strip())
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalize_datetime(value: Any) -> Optional[str]:
    """-> 'YYYY-MM-DD HH:MM' (seconds dropped), or 'YYYY-MM-DD' when no time is present."""
    if not isinstance(value, str):
        return None
    v = re.sub(r"\s+", " ", value.strip().replace("T", " "))
    v = re.sub(r"(?i)\b(\d{1,2}:\d{2}(?::\d{2})?)\s*([ap]m)\b", lambda m: f"{m[1]} {m[2].upper()}", v)
    only_date = normalize_date(v)
    if only_date:
        return only_date
    # split "<date part> <time part>" on the last time-looking token
    m = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s*$", v)
    if not m:
        return None
    date_part = normalize_date(v[: m.start()].strip(" ,"))
    if not date_part:
        return None
    time_raw = m.group(1)
    for fmt in TIME_FORMATS:
        try:
            return f"{date_part} {datetime.strptime(time_raw, fmt).strftime('%H:%M')}"
        except ValueError:
            continue
    return None


def _clean_str(v: str) -> Optional[str]:
    v = v.strip()
    return v or None


def normalize_records(doc_type: str, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Issue]]:
    out: List[Dict[str, Any]] = []
    issues: List[Issue] = []
    for idx, rec in enumerate(records):
        clean: Dict[str, Any] = {}
        for key, val in rec.items():
            if isinstance(val, str):
                val = _clean_str(val)
            elif isinstance(val, list):
                val = [s.strip() for s in val if isinstance(s, str) and s.strip()]
            elif isinstance(val, float):
                val = round(val, 2)

            if val and key in DATE_FIELDS:
                iso = normalize_date(val)
                if iso:
                    val = iso
                else:
                    issues.append(Issue("warning", key, f"could not parse date {val!r}", idx))
            elif val and key in DATETIME_FIELDS:
                iso = normalize_datetime(val)
                if iso:
                    val = iso
                else:
                    issues.append(Issue("warning", key, f"could not parse date/time {val!r}", idx))
            elif key == "currency" and isinstance(val, str):
                val = CURRENCY_SYMBOLS.get(val.upper(), CURRENCY_SYMBOLS.get(val, val)).upper()
            clean[key] = val
        out.append(clean)
    return out, issues


# ------------------------------------------------------------------ checks
def _close(a: float, b: float, rel: float = 0.005, abs_tol: float = 0.05) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def _check_currency(rec: Dict[str, Any], idx: int) -> List[Issue]:
    cur = rec.get("currency")
    if cur is None:
        return [Issue("warning", "currency", "missing", idx)]
    if not re.fullmatch(r"[A-Z]{3}", cur):
        return [Issue("warning", "currency", f"{cur!r} is not an ISO 4217 code", idx)]
    return []


def _validate_invoice(rec: Dict[str, Any], idx: int) -> List[Issue]:
    issues: List[Issue] = []
    sub, tax, total = rec.get("subtotal"), rec.get("tax_amount"), rec.get("total_due")
    if total is None:
        issues.append(Issue("error", "total_due", "missing; re-read the amount payable", idx))
    elif total <= 0:
        issues.append(Issue("error", "total_due", f"must be positive, got {total}", idx))
    if tax is not None and tax < 0:
        issues.append(Issue("error", "tax_amount", f"must not be negative, got {tax}", idx))
    if tax is not None and total is not None and tax > total:
        issues.append(Issue("error", "tax_amount", f"tax {tax} exceeds total {total}", idx))
    if sub is not None and tax is not None and total is not None and not _close(sub + tax, total):
        issues.append(
            Issue(
                "error", "total_due", f"subtotal {sub} + tax {tax} = {round(sub + tax, 2)}, but total is {total}", idx
            )
        )
    for f in ("invoice_number", "vendor_name", "issue_date"):
        if rec.get(f) is None:
            issues.append(Issue("warning", f, "missing", idx))
    return issues + _check_currency(rec, idx)


def _validate_receipt(rec: Dict[str, Any], idx: int) -> List[Issue]:
    issues: List[Issue] = []
    sub, tax, total = rec.get("subtotal"), rec.get("tax_amount"), rec.get("total_paid")
    if total is None:
        issues.append(Issue("error", "total_paid", "missing; re-read the amount paid", idx))
    elif total <= 0:
        issues.append(Issue("error", "total_paid", f"must be positive, got {total}", idx))
    # Tips, discounts and rounding are common on receipts, so arithmetic mismatches are warnings only.
    if sub is not None and total is not None and sub > total + 0.05:
        issues.append(Issue("warning", "subtotal", f"subtotal {sub} exceeds total {total}", idx))
    if sub is not None and tax is not None and total is not None and not _close(sub + tax, total):
        issues.append(Issue("warning", "total_paid", f"subtotal {sub} + tax {tax} != total {total}", idx))
    if not rec.get("items_bought"):
        issues.append(Issue("warning", "items_bought", "no items found", idx))
    if rec.get("store_name") is None:
        issues.append(Issue("warning", "store_name", "missing", idx))
    return issues + _check_currency(rec, idx)


def validate_records(doc_type: str, records: List[Dict[str, Any]]) -> List[Issue]:
    if not records:
        return [Issue("error", "items", "no records were extracted; re-read the whole document", 0)]
    fn = {"invoice": _validate_invoice, "receipt": _validate_receipt}.get(doc_type)
    if fn is None:
        return []
    issues: List[Issue] = []
    for i, rec in enumerate(records):
        issues.extend(fn(rec, i))
    return issues


def format_feedback(issues: List[Issue]) -> str:
    return "\n".join(f"- {i}" for i in issues if i.severity == "error")
