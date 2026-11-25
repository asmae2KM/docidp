"""Streamlit UI:  streamlit run app.py"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from docidp.config import Settings
from docidp.errors import DocIDPError
from docidp.export import records_to_dataframe, results_to_excel, results_to_json
from docidp.llm import LangChainBackend
from docidp.parsers import SUPPORTED_EXTS, load_document
from docidp.pipeline import DocumentPipeline

load_dotenv()
st.set_page_config(page_title="Document extractor", layout="wide")
st.title("Document extractor")
st.caption(
    "PDFs and images → routed → schema-validated records → Excel / JSON. Low-confidence results are flagged for review."
)

try:
    settings = Settings.from_env()
except DocIDPError as exc:
    st.error(str(exc))
    st.stop()


@st.cache_resource
def get_backend(_settings: Settings) -> LangChainBackend:
    return LangChainBackend(_settings)


# Results live in session_state so they survive Streamlit reruns (e.g. clicking a download button).
st.session_state.setdefault("results", [])
st.session_state.setdefault("errors", [])

with st.sidebar:
    st.header("Input")
    files = st.file_uploader(
        "Upload PDFs or images", type=[e.lstrip(".") for e in SUPPORTED_EXTS], accept_multiple_files=True
    )
    instruction = st.text_area("Optional guidance", placeholder="e.g. focus on totals and tax", max_chars=500)
    max_repairs = st.slider(
        "Max repair attempts", 0, 4, settings.max_repairs, help="Re-ask the model when consistency checks fail."
    )
    run = st.button("Extract", type="primary", disabled=not files)
    st.divider()
    st.caption(f"Text: `{settings.router_model}` / `{settings.extractor_model}` ({settings.text_provider})")
    st.caption(f"Images: `{settings.vision_model}` ({settings.vision_provider})")
    st.caption("Uploaded documents are sent to the configured LLM providers.")

if run and files:
    cfg = settings.with_overrides(max_repairs=max_repairs)
    pipeline = DocumentPipeline(get_backend(settings), cfg)
    st.session_state.results, st.session_state.errors = [], []
    bar = st.progress(0.0)
    for i, f in enumerate(files):
        with st.spinner(f"Processing {f.name}..."):
            try:
                doc = load_document(f.getvalue(), f.name, cfg)
                st.session_state.results.append(pipeline.run(doc, instruction))
            except DocIDPError as exc:
                st.session_state.errors.append((f.name, str(exc)))
        bar.progress((i + 1) / len(files))
    bar.empty()

results, errors = st.session_state.results, st.session_state.errors

for name, msg in errors:
    st.error(f"**{name}**: {msg}")

if results:
    st.subheader("Summary")
    st.dataframe(
        pd.DataFrame(
            {
                "file": r.source,
                "type": r.document_type,
                "confidence": round(r.route.confidence, 2),
                "records": len(r.records),
                "attempts": r.attempts,
                "needs review": r.needs_review,
                "seconds": round(r.timings["total_s"], 1),
            }
            for r in results
        ),
        hide_index=True,
    )
    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button(
        "Download Excel", results_to_excel(results), "extraction.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )  # fmt: skip
    c2.download_button("Download JSON", results_to_json(results), "extraction.json", "application/json")

    for r in results:
        badge = "⚠️ review" if r.needs_review else "✅"
        with st.expander(
            f"{badge}  {r.source}: {r.document_type.upper()} ({len(r.records)} record(s))", expanded=r.needs_review
        ):
            st.caption(f"Router: {r.route.reasoning} (confidence {r.route.confidence:.2f}, self-reported)")
            if r.review_reasons:
                st.warning("; ".join(r.review_reasons))
            if r.attempts > 1:
                st.info(f"Consistency checks triggered {r.attempts - 1} repair attempt(s).")
            st.dataframe(records_to_dataframe(r.document_type, r.records), hide_index=True)
            if r.issues:
                st.markdown("**Validation**")
                for issue in r.issues:
                    (st.error if issue.severity == "error" else st.warning)(str(issue))
else:
    st.info("Upload one or more documents in the sidebar and press **Extract**.")
