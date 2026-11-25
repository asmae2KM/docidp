"""Runs the full evaluation harness over the real generated files with an oracle backend."""

import json

from docidp.config import Settings
from evaluation.generate_dataset import generate
from evaluation.run_eval import README_END, README_START, render_markdown, run_evaluation, summarize, update_readme

from .fakes import OracleBackend

SETTINGS = Settings(pdf_parser="pypdf")


def test_dataset_is_deterministic(tmp_path):
    a = generate(tmp_path / "a", seed=7)
    b = generate(tmp_path / "b", seed=7)
    assert a == b
    assert a != generate(tmp_path / "c", seed=8)


def test_dataset_covers_the_hard_cases(dataset_dir):
    docs = json.loads((dataset_dir / "labels.json").read_text())["documents"]
    tags = {t for d in docs for t in d["tags"]}
    assert {"multi-record", "prompt-injection", "hard-negative", "noisy-image"} <= tags
    assert {d["modality"] for d in docs} == {"text", "image"}
    assert {d["expected_type"] for d in docs} == {"invoice", "receipt", "general"}
    assert len(docs) >= 40


def test_oracle_scores_perfectly_on_every_real_file(dataset_dir):
    rows = run_evaluation(dataset_dir, OracleBackend(dataset_dir), SETTINGS)
    s = summarize(rows)
    assert s["pipeline_failures"] == 0, s["failures"]
    assert s["routing_accuracy_pct"] == 100.0
    assert s["field_accuracy_pct"] == 100.0 and s["doc_exact_match_pct"] == 100.0
    assert s["review_flag"]["flagged"] == 0


def test_summary_measures_repair_effect_and_review_flag(dataset_dir):
    corrupt = {"inv-001.pdf": {"total_due": 1.0}}  # first attempt wrong, repair fixes it
    be = OracleBackend(dataset_dir, corrupt_first=corrupt)
    s = summarize(run_evaluation(dataset_dir, be, SETTINGS))
    assert s["repair"]["triggered_docs"] == 1 and s["repair"]["fixed"] == 1 and s["repair"]["regressed"] == 0
    assert s["repair"]["exact_first_attempt_pct"] < s["repair"]["exact_final_pct"] == 100.0
    assert s["field_accuracy_first_attempt_pct"] < s["field_accuracy_pct"]


def test_review_flag_recall_with_undetectable_error(dataset_dir):
    # a wrong vendor name passes every arithmetic check, so it must show up as a missed error
    be = OracleBackend(dataset_dir, corrupt_first={"inv-002.pdf": {"vendor_name": "Someone Else"}})
    s = summarize(run_evaluation(dataset_dir, be, SETTINGS))
    assert s["review_flag"]["wrong_docs"] == 1 and s["review_flag"]["recall_pct"] == 0.0


def test_modality_filter_and_limit(dataset_dir):
    rows = run_evaluation(dataset_dir, OracleBackend(dataset_dir), SETTINGS, modality="image", limit=3)
    assert len(rows) == 3 and {r["modality"] for r in rows} == {"image"}


def test_pipeline_failures_count_against_the_score(dataset_dir):
    class Broken(OracleBackend):
        def route(self, doc):
            from docidp.errors import ExtractionError

            raise ExtractionError("boom")

    s = summarize(run_evaluation(dataset_dir, Broken(dataset_dir), SETTINGS, limit=4))
    assert s["pipeline_failures"] == 4 and s["routing_accuracy_pct"] == 0.0


def test_markdown_report_and_readme_update(dataset_dir, tmp_path):
    rows = run_evaluation(dataset_dir, OracleBackend(dataset_dir), SETTINGS, limit=10)
    meta = {"timestamp": "t", "dataset": "synthetic", "text_models": "a / b", "vision_model": "v", "llm_calls": 5}
    md = render_markdown(summarize(rows), meta)
    assert "Routing accuracy" in md and "| `total_due` |" in md
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\noutro")
    update_readme(readme, md)
    text = readme.read_text()
    assert "old" not in text and "Routing accuracy" in text and text.startswith("intro") and text.endswith("outro")


def test_delay_is_applied_between_documents_only(dataset_dir, monkeypatch):
    sleeps = []
    monkeypatch.setattr("evaluation.run_eval.time.sleep", sleeps.append)
    run_evaluation(dataset_dir, OracleBackend(dataset_dir), SETTINGS, limit=4, delay=2.5)
    assert sleeps == [2.5, 2.5, 2.5]
