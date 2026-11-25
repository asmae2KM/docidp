from io import BytesIO

from openpyxl import load_workbook

from docidp.export import records_to_dataframe, results_to_excel, results_to_json
from docidp.pipeline import ExtractionResult
from docidp.schemas import RouteDecision
from docidp.validators import Issue


def result(doc_type, records, issues=(), review=False):
    return ExtractionResult(
        source="f.pdf", route=RouteDecision(document_type=doc_type, confidence=0.9, reasoning="r"),
        records=records, issues=list(issues), attempts=1, needs_review=review,
        review_reasons=["x"] if review else [], timings={"total_s": 1.0},
    )  # fmt: skip


RECEIPT = {
    "store_name": "Shop", "transaction_time": "2026-01-01 10:00", "items_bought": ["milk", "bread"],
    "subtotal": 5.0, "tax_amount": None, "total_paid": 5.0, "currency": "EUR",
}  # fmt: skip


def test_list_fields_are_flattened():
    """Regression: v1 passed list cells to openpyxl and crashed on every receipt export."""
    df = records_to_dataframe("receipt", [RECEIPT])
    assert df.loc[0, "items_bought"] == "milk; bread"


def test_excel_roundtrip_has_expected_sheets_and_data():
    data = results_to_excel([result("receipt", [RECEIPT], [Issue("warning", "currency", "x")], review=True)])
    wb = load_workbook(BytesIO(data))
    assert wb.sheetnames == ["Summary", "Receipts", "Validation"]
    rows = list(wb["Receipts"].values)
    assert rows[0][:3] == ("source_file", "record_no", "needs_review")
    assert "milk; bread" in rows[1]
    assert list(wb["Validation"].values)[1][1] == "warning"


def test_formula_like_text_is_stored_as_text():
    evil = {**RECEIPT, "store_name": '=HYPERLINK("http://evil","x")'}
    wb = load_workbook(BytesIO(results_to_excel([result("receipt", [evil])])))
    cell = wb["Receipts"]["D2"]
    assert cell.data_type == "s" and cell.value.startswith("=")


def test_empty_results_still_produce_a_workbook():
    wb = load_workbook(BytesIO(results_to_excel([])))
    assert "Summary" in wb.sheetnames


def test_json_export_is_valid():
    import json

    out = json.loads(results_to_json([result("receipt", [RECEIPT])]))
    assert out[0]["document_type"] == "receipt" and out[0]["records"][0]["store_name"] == "Shop"
