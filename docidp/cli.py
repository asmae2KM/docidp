"""Command line interface:  docidp extract invoices/ -o report.xlsx --json report.json"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from .config import Settings
from .errors import DocIDPError
from .export import results_to_excel, results_to_json
from .llm import LangChainBackend
from .parsers import SUPPORTED_EXTS, load_document
from .pipeline import DocumentPipeline, ExtractionResult


def _collect(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTS))
        else:
            files.append(p)
    return files


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="docidp", description="Extract structured data from PDFs and images.")
    sub = ap.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract", help="extract one or more files / folders")
    ex.add_argument("paths", nargs="+")
    ex.add_argument("-o", "--excel", help="write an .xlsx report here")
    ex.add_argument("--json", dest="json_out", help="write full JSON results here")
    ex.add_argument("-i", "--instruction", default="", help="optional extraction guidance")
    ex.add_argument("--max-repairs", type=int, help="override MAX_REPAIRS")
    ex.add_argument("--delay", type=float, default=0.0, help="seconds to wait between files (free-tier rate limits)")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.max_repairs is not None:
            settings = settings.with_overrides(max_repairs=max(0, args.max_repairs))
    except DocIDPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pipeline = DocumentPipeline(LangChainBackend(settings), settings)
    results: List[ExtractionResult] = []
    failures = 0
    for n, path in enumerate(_collect(args.paths)):
        if n and args.delay > 0:
            time.sleep(args.delay)
        try:
            doc = load_document(path.read_bytes(), path.name, settings)
            res = pipeline.run(doc, args.instruction)
        except (DocIDPError, OSError) as exc:
            failures += 1
            print(f"FAIL  {path.name}: {exc}", file=sys.stderr)
            continue
        results.append(res)
        flag = "REVIEW" if res.needs_review else "ok"
        print(
            f"{flag:<6} {path.name:<32} {res.document_type:<8} conf={res.route.confidence:.2f} "
            f"records={len(res.records)} attempts={res.attempts} {res.timings['total_s']:.1f}s"
        )

    if results and args.excel:
        Path(args.excel).write_bytes(results_to_excel(results))
        print(f"wrote {args.excel}")
    if results and args.json_out:
        Path(args.json_out).write_text(results_to_json(results), encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
