from docidp.config import Settings
from docidp.parsers import Document
from docidp.pipeline import DocumentPipeline
from docidp.schemas import RouteDecision

from .fakes import ScriptedBackend

DOC = Document(name="a.pdf", kind="text", content="irrelevant for the fake backend", parse_seconds=0.1)
ROUTE = RouteDecision(document_type="invoice", confidence=0.95, reasoning="has an invoice number")

GOOD = {"invoice_number": "A1", "issue_date": "14/03/2026", "vendor_name": "ACME", "currency": "eur",
        "subtotal": 100.0, "tax_amount": 20.0, "total_due": 120.0}  # fmt: skip
BAD = {**GOOD, "total_due": 210.0}  # misread digits: 120 -> 210


def pipe(backend, **kw):
    return DocumentPipeline(backend, Settings(**kw))


def test_clean_extraction_needs_one_call_and_no_review():
    be = ScriptedBackend(ROUTE, [[GOOD]])
    res = pipe(be).run(DOC)
    assert res.attempts == 1 and not res.needs_review and be.calls == 2
    assert res.records[0]["issue_date"] == "2026-03-14" and res.records[0]["currency"] == "EUR"  # normalised


def test_repair_loop_recovers_and_passes_feedback_to_the_model():
    be = ScriptedBackend(ROUTE, [[BAD], [GOOD]])
    res = pipe(be).run(DOC)
    assert res.attempts == 2 and not res.needs_review
    assert res.records[0]["total_due"] == 120.0
    assert be.feedbacks[0] is None and "subtotal 100.0 + tax 20.0" in be.feedbacks[1]
    assert res.history[0].error_count == 1 and res.history[1].error_count == 0


def test_gives_up_after_max_repairs_and_flags_for_review():
    be = ScriptedBackend(ROUTE, [[BAD]])
    res = pipe(be, max_repairs=2).run(DOC)
    assert res.attempts == 3 and res.needs_review
    assert "validation error" in res.review_reasons[0]


def test_max_repairs_zero_disables_the_loop():
    res = pipe(ScriptedBackend(ROUTE, [[BAD]]), max_repairs=0).run(DOC)
    assert res.attempts == 1 and res.needs_review


def test_best_attempt_is_kept_and_a_repair_cannot_make_things_worse():
    worse = {**BAD, "tax_amount": -5.0}  # two errors
    res = pipe(ScriptedBackend(ROUTE, [[BAD], [worse], [worse]])).run(DOC)
    assert res.records[0]["tax_amount"] == 20.0 and res.history[0].attempt == 1


def test_low_router_confidence_is_flagged_even_when_extraction_is_clean():
    low = RouteDecision(document_type="invoice", confidence=0.3, reasoning="unsure")
    res = pipe(ScriptedBackend(low, [[GOOD]])).run(DOC)
    assert res.needs_review and "low routing confidence" in res.review_reasons[0]


def test_truncated_documents_are_flagged():
    doc = Document(name="a.pdf", kind="text", content="x", truncated=True)
    assert "truncated" in pipe(ScriptedBackend(ROUTE, [[GOOD]])).run(doc).review_reasons[0]


def test_empty_extraction_triggers_repair():
    res = pipe(ScriptedBackend(ROUTE, [[], [GOOD]])).run(DOC)
    assert res.attempts == 2 and len(res.records) == 1


def test_multi_record_documents_are_supported():
    second = {**GOOD, "invoice_number": "A2"}
    res = pipe(ScriptedBackend(ROUTE, [[GOOD, second]])).run(DOC)
    assert [r["invoice_number"] for r in res.records] == ["A1", "A2"]


def test_result_serialises_to_dict():
    d = pipe(ScriptedBackend(ROUTE, [[GOOD]])).run(DOC).to_dict()
    assert d["document_type"] == "invoice" and d["timings_s"]["parse_s"] == 0.1
