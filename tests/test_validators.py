import pytest

from docidp.validators import (
    format_feedback,
    normalize_date,
    normalize_datetime,
    normalize_records,
    validate_records,
)


@pytest.mark.parametrize(
    "raw,iso",
    [
        ("2026-03-14", "2026-03-14"),
        ("14/03/2026", "2026-03-14"),
        ("14.03.2026", "2026-03-14"),
        ("14 March 2026", "2026-03-14"),
        ("March 14, 2026", "2026-03-14"),
        ("03/14/2026", "2026-03-14"),  # only valid month-first
        ("05/04/2026", "2026-04-05"),  # ambiguous -> day-first, documented behaviour
        ("  14   March  2026 ", "2026-03-14"),
    ],
)
def test_normalize_date(raw, iso):
    assert normalize_date(raw) == iso


@pytest.mark.parametrize("raw", ["", "soon", "2026-13-45", None, 5])
def test_normalize_date_rejects_garbage(raw):
    assert normalize_date(raw) is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-08-10 14:22", "2026-08-10 14:22"),
        ("2026-08-10 14:22:05", "2026-08-10 14:22"),
        ("2026-08-10T14:22:05", "2026-08-10 14:22"),
        ("10/08/2026 14:22", "2026-08-10 14:22"),
        ("Aug 10 2026 2:22 PM", "2026-08-10 14:22"),
        ("Aug 10 2026 12:05 AM", "2026-08-10 00:05"),
        ("2026-08-10", "2026-08-10"),
    ],
)
def test_normalize_datetime(raw, expected):
    assert normalize_datetime(raw) == expected


def test_normalize_records_cleans_values():
    recs, issues = normalize_records(
        "invoice",
        [
            {
                "issue_date": "14/03/2026",
                "currency": " € ",
                "vendor_name": "  ACME ",
                "total_due": 10.004,
                "tax_amount": None,
            }
        ],
    )
    assert recs[0]["issue_date"] == "2026-03-14"
    assert recs[0]["currency"] == "EUR"
    assert recs[0]["vendor_name"] == "ACME"
    assert recs[0]["total_due"] == 10.0
    assert issues == []


def test_normalize_records_flags_unparseable_date():
    _, issues = normalize_records("invoice", [{"issue_date": "someday"}])
    assert [i.severity for i in issues] == ["warning"]


GOOD_INVOICE = {
    "invoice_number": "A1", "issue_date": "2026-01-01", "vendor_name": "ACME", "currency": "EUR",
    "subtotal": 100.0, "tax_amount": 20.0, "total_due": 120.0,
}  # fmt: skip


def errors(issues):
    return [i for i in issues if i.severity == "error"]


def test_valid_invoice_has_no_errors():
    assert errors(validate_records("invoice", [GOOD_INVOICE])) == []


def test_invoice_arithmetic_mismatch_is_an_error():
    bad = {**GOOD_INVOICE, "total_due": 210.0}
    errs = errors(validate_records("invoice", [bad]))
    assert len(errs) == 1 and "subtotal 100.0 + tax 20.0" in errs[0].message
    assert "total_due" in format_feedback(errs)


def test_invoice_rounding_tolerance():
    assert errors(validate_records("invoice", [{**GOOD_INVOICE, "total_due": 120.03}])) == []


def test_invoice_missing_total_and_empty_records_are_errors():
    assert errors(validate_records("invoice", [{**GOOD_INVOICE, "total_due": None}]))
    assert errors(validate_records("invoice", []))


def test_tax_greater_than_total_is_an_error():
    assert errors(validate_records("invoice", [{**GOOD_INVOICE, "subtotal": None, "tax_amount": 500.0}]))


def test_currency_symbol_is_only_a_warning():
    issues = validate_records("invoice", [{**GOOD_INVOICE, "currency": "$"}])
    assert errors(issues) == [] and any(i.field == "currency" for i in issues)


def test_receipt_mismatch_is_warning_not_error():
    rec = {
        "store_name": "S",
        "items_bought": ["x"],
        "subtotal": 10.0,
        "tax_amount": 1.0,
        "total_paid": 15.0,
        "currency": "MAD",
    }
    issues = validate_records("receipt", [rec])
    assert errors(issues) == [] and any(i.field == "total_paid" for i in issues)


def test_general_documents_need_no_numeric_checks():
    assert validate_records("general", [{"document_title": "X"}]) == []
