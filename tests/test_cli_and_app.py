import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from docidp import cli

from .fakes import OracleBackend

ROOT = Path(__file__).resolve().parents[1]


def test_cli_extracts_a_folder_to_excel_and_json(dataset_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PDF_PARSER", "pypdf")
    monkeypatch.setattr(cli, "LangChainBackend", lambda settings: OracleBackend(dataset_dir))
    xlsx, js = tmp_path / "out.xlsx", tmp_path / "out.json"
    sample = tmp_path / "in"
    sample.mkdir()
    for name in ("inv-001.pdf", "rec-001.jpg", "gen-001.pdf"):
        (sample / name).write_bytes((dataset_dir / "files" / name).read_bytes())
    code = cli.main(["extract", str(sample), "-o", str(xlsx), "--json", str(js)])
    assert code == 0 and xlsx.stat().st_size > 0
    out = json.loads(js.read_text())
    assert {r["document_type"] for r in out} == {"invoice", "receipt", "general"}
    assert "inv-001.pdf" in capsys.readouterr().out


def test_cli_reports_failures_with_nonzero_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "LangChainBackend", lambda settings: None)
    bad = tmp_path / "notes.txt"
    bad.write_text("hi")
    assert cli.main(["extract", str(bad)]) == 1
    assert "Unsupported file type" in capsys.readouterr().err


def test_cli_rejects_invalid_config(monkeypatch, capsys):
    monkeypatch.setenv("TEXT_PROVIDER", "nope")
    assert cli.main(["extract", "x.pdf"]) == 2


def test_streamlit_app_boots_without_errors():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Document extractor"
    assert any("Upload one or more documents" in i.value for i in at.info)


def test_cli_delay_between_files(dataset_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_PARSER", "pypdf")
    monkeypatch.setattr(cli, "LangChainBackend", lambda settings: OracleBackend(dataset_dir))
    sleeps = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    sample = tmp_path / "in"
    sample.mkdir()
    for name in ("inv-001.pdf", "inv-002.pdf", "gen-001.pdf"):
        (sample / name).write_bytes((dataset_dir / "files" / name).read_bytes())
    assert cli.main(["extract", str(sample), "--delay", "1.5"]) == 0
    assert sleeps == [1.5, 1.5]
