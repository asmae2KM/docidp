import pytest

from evaluation.metrics import list_f1, score_document, score_field


@pytest.mark.parametrize(
    "name,gold,pred,expected",
    [
        ("total_due", 1920.0, 1920.004, 1.0),
        ("total_due", 1920.0, 1920.5, 0.0),
        ("total_due", 1920.0, None, 0.0),
        ("total_due", 1920.0, "abc", 0.0),
        ("tax_amount", None, None, 1.0),  # no tax line: null or 0 both fine
        ("tax_amount", None, 0, 1.0),
        ("tax_amount", None, 14.5, 0.0),
        ("issue_date", "2026-03-14", "14/03/2026", 1.0),
        ("issue_date", "2026-03-14", "2026-03-15", 0.0),
        ("transaction_time", "2026-08-10 14:22", "2026-08-10 14:22:59", 1.0),
        ("currency", "EUR", "eur", 1.0),
        ("invoice_number", "INV-2026-104", "inv 2026 104", 1.0),
        ("invoice_number", "INV-2026-104", "INV-2026-105", 0.0),
        ("vendor_name", "Atlas Industrial Supplies", "Atlas Industrial Supplies.", 1.0),
        ("vendor_name", "Atlas Industrial Supplies", "Atlas Industrial", 0.0),
        ("store_name", "Cafe Paul Rabat", None, 0.0),
    ],
)
def test_score_field(name, gold, pred, expected):
    assert score_field(name, gold, pred) == expected


def test_list_f1():
    assert list_f1(["Olive Oil 1L", "Water"], ["olive oil 1l", "water"]) == 1.0
    assert list_f1(["a1", "b2"], ["a1"]) == pytest.approx(2 / 3)
    assert list_f1(["a1"], []) == 0.0
    assert list_f1([], []) == 1.0


def test_matching_handles_reordered_multi_record_documents():
    gold = [{"invoice_number": "A-1", "total_due": 10.0}, {"invoice_number": "B-2", "total_due": 20.0}]
    pred = [{"invoice_number": "B-2", "total_due": 20.0}, {"invoice_number": "A-1", "total_due": 10.0}]
    assert score_document("invoice", gold, pred).exact


def test_missing_and_extra_records_are_penalised():
    gold = [{"invoice_number": "A-1", "total_due": 10.0}, {"invoice_number": "B-2", "total_due": 20.0}]
    missing = score_document("invoice", gold, [gold[0]])
    assert not missing.exact and missing.accuracy == 0.5
    extra = score_document("invoice", gold[:1], gold)
    assert not extra.exact and extra.accuracy == 1.0


def test_single_record_wrong_id_is_a_wrong_field_not_a_missing_record():
    s = score_document(
        "invoice", [{"invoice_number": "A-1", "total_due": 10.0}], [{"invoice_number": "ZZZ", "total_due": 10.0}]
    )
    assert s.accuracy == 0.5 and s.n_pred == 1
