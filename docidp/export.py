"""Excel / JSON export. Regression-safe: list fields are flattened (v1 crashed on them)."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Dict, List, Sequence

import pandas as pd
from openpyxl.styles import Font

from .pipeline import ExtractionResult
from .schemas import record_fields

SHEET_NAMES = {"invoice": "Invoices", "receipt": "Receipts", "general": "General"}


def _flatten(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    return value


def records_to_dataframe(doc_type: str, records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    cols = record_fields(doc_type)
    rows = [{c: _flatten(rec.get(c)) for c in cols} for rec in records]
    return pd.DataFrame(rows, columns=cols)


def _combined_frame(doc_type: str, results: Sequence[ExtractionResult]) -> pd.DataFrame:
    frames = []
    for r in results:
        if r.document_type != doc_type or not r.records:
            continue
        df = records_to_dataframe(doc_type, r.records)
        df.insert(0, "needs_review", r.needs_review)
        df.insert(0, "record_no", range(1, len(df) + 1))
        df.insert(0, "source_file", r.source)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _finish_sheet(ws) -> None:
    ws.freeze_panes = "A2"
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for col in ws.columns:
        longest = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[col[0].column_letter].width = min(60, longest + 2)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            # Text from untrusted documents must never be evaluated as an Excel formula.
            if isinstance(cell.value, str) and cell.value.startswith("="):
                cell.data_type = "s"


def results_to_excel(results: Sequence[ExtractionResult]) -> bytes:
    summary = pd.DataFrame(
        [
            {
                "source_file": r.source,
                "document_type": r.document_type,
                "router_confidence": round(r.route.confidence, 2),
                "records": len(r.records),
                "attempts": r.attempts,
                "needs_review": r.needs_review,
                "review_reasons": "; ".join(r.review_reasons),
                "seconds": round(r.timings.get("total_s", 0.0), 2),
            }
            for r in results
        ]
    )
    issue_rows: List[Dict[str, Any]] = [
        {
            "source_file": r.source,
            "severity": i.severity,
            "record": i.record_index + 1,
            "field": i.field,
            "message": i.message,
        }
        for r in results
        for i in r.issues
    ]
    issues = pd.DataFrame(issue_rows, columns=["source_file", "severity", "record", "field", "message"])

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        for doc_type, sheet in SHEET_NAMES.items():
            df = _combined_frame(doc_type, results)
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet, index=False)
        issues.to_excel(writer, sheet_name="Validation", index=False)
        for ws in writer.book.worksheets:
            _finish_sheet(ws)
    return buf.getvalue()


def results_to_json(results: Sequence[ExtractionResult]) -> str:
    return json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False)
