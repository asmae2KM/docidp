"""Data contracts.

Every business field is *required but nullable*: the model must emit the key, and is
allowed (and told) to emit null when the document does not contain the value. The v1
schemas used non-nullable fields, which pushed the model to hallucinate missing values.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Type

from pydantic import BaseModel, Field, field_validator

DocumentType = Literal["invoice", "receipt", "general"]


class RouteDecision(BaseModel):
    document_type: DocumentType = Field(description="Category of the document.")
    confidence: float = Field(description="Self-assessed confidence between 0 and 1.")
    reasoning: str = Field(description="One short sentence explaining the choice.")

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v):
        # Models sometimes answer 85 instead of 0.85, or null. Normalise instead of failing.
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.5
        if 1 < v <= 100:
            v = v / 100
        return min(1.0, max(0.0, v))


# ---------------------------------------------------------------- invoice
class InvoiceRecord(BaseModel):
    invoice_number: Optional[str] = Field(description="Invoice / facture / Rechnung number exactly as printed.")
    issue_date: Optional[str] = Field(description="Issue date, ISO format YYYY-MM-DD.")
    vendor_name: Optional[str] = Field(description="Company that ISSUED the invoice (the seller, not the customer).")
    currency: Optional[str] = Field(description="ISO 4217 code (EUR, USD, MAD...). Infer from symbols if needed.")
    subtotal: Optional[float] = Field(description="Total before tax (net / HT / Netto).")
    tax_amount: Optional[float] = Field(description="Total tax amount (VAT / TVA / MwSt / sales tax).")
    total_due: Optional[float] = Field(description="Final amount payable (gross / TTC / Brutto / balance due).")


class InvoiceExtraction(BaseModel):
    items: List[InvoiceRecord] = Field(description="One record per distinct invoice in the document.")


# ---------------------------------------------------------------- receipt
class ReceiptRecord(BaseModel):
    store_name: Optional[str] = Field(description="Store or restaurant name.")
    transaction_time: Optional[str] = Field(description="Purchase date and time, ISO format YYYY-MM-DD HH:MM.")
    items_bought: List[str] = Field(description="Names of purchased items only (no prices, no quantities).")
    subtotal: Optional[float] = Field(description="Total before tax.")
    tax_amount: Optional[float] = Field(description="Tax amount, or null if the receipt prints no tax line.")
    total_paid: Optional[float] = Field(description="Final amount paid.")
    currency: Optional[str] = Field(description="ISO 4217 code printed on or implied by the receipt.")


class ReceiptExtraction(BaseModel):
    items: List[ReceiptRecord] = Field(description="One record per distinct receipt in the document.")


# ---------------------------------------------------------------- general
class GeneralRecord(BaseModel):
    document_title: Optional[str] = Field(description="Title or headline printed on the document.")
    creation_date: Optional[str] = Field(description="Main date on the document, ISO format YYYY-MM-DD.")
    key_entities: List[str] = Field(description="Important people and organisations named.")
    summary_metrics: Optional[str] = Field(description="Key figures or amounts, in one short string.")


class GeneralExtraction(BaseModel):
    items: List[GeneralRecord] = Field(description="Exactly one record describing the document.")


SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "invoice": InvoiceExtraction,
    "receipt": ReceiptExtraction,
    "general": GeneralExtraction,
}
RECORD_MODELS: Dict[str, Type[BaseModel]] = {
    "invoice": InvoiceRecord,
    "receipt": ReceiptRecord,
    "general": GeneralRecord,
}


def record_fields(doc_type: str) -> List[str]:
    """Stable, schema-ordered column names for a document type."""
    return list(RECORD_MODELS[doc_type].model_fields)
