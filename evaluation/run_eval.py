"""Reproducible evaluation of the whole pipeline (parsing -> routing -> extraction -> validation).

    python -m evaluation.run_eval                       # needs GROQ_API_KEY + OPENAI_API_KEY
    python -m evaluation.run_eval --modality text       # PDFs only (Groq key only)
    python -m evaluation.run_eval --update-readme       # refresh the README results block

Everything is measured on the same run: attempt 1 (no repair loop) vs the final result, so the
effect of the validation/repair loop comes for free, with no second pass over the API.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from docidp.config import Settings
from docidp.errors import DocIDPError
from docidp.llm import LLMBackend
from docidp.parsers import load_document
from docidp.pipeline import DocumentPipeline

from .metrics import score_document

README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


# ------------------------------------------------------------------ running
def run_evaluation(
    dataset_dir: Path, backend: LLMBackend, settings: Settings, modality: str = "all", limit: Optional[int] = None,
    progress: bool = False,
    delay: float = 0.0,
) -> List[Dict[str, Any]]:  # fmt: skip
    labels = json.loads((dataset_dir / "labels.json").read_text(encoding="utf-8"))
    docs = [d for d in labels["documents"] if modality in ("all", d["modality"])]
    if limit:
        docs = docs[:limit]
    pipeline = DocumentPipeline(backend, settings)
    rows: List[Dict[str, Any]] = []
    for i, d in enumerate(docs, 1):
        if i > 1 and delay > 0:
            time.sleep(delay)  # stay under per-minute token caps on free tiers
        path = dataset_dir / d["file"]
        row: Dict[str, Any] = {
            "id": d["id"], "modality": d["modality"], "expected_type": d["expected_type"], "tags": d["tags"],
            "error": None,
        }  # fmt: skip
        try:
            doc = load_document(path.read_bytes(), path.name, settings)
            res = pipeline.run(doc)
            gold = d["records"]
            final = score_document(d["expected_type"], gold, res.records)
            first = score_document(d["expected_type"], gold, res.history[0].records)
            row.update(
                predicted_type=res.document_type,
                routed_ok=res.document_type == d["expected_type"],
                confidence=res.route.confidence,
                attempts=res.attempts,
                needs_review=res.needs_review,
                has_gold=bool(gold),
                exact=final.exact,
                exact_first=first.exact,
                fields=final.fields,
                fields_first=first.fields,
                timings=res.timings,
            )
        except DocIDPError as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if progress:
            status = row["error"] or f"{row['predicted_type']:<8} exact={row['exact']} attempts={row['attempts']}"
            print(f"[{i:>2}/{len(docs)}] {d['id']:<8} {status}", flush=True)
    return rows


# ------------------------------------------------------------------ summarising
def _pct(n: float, d: float) -> Optional[float]:
    return round(100 * n / d, 1) if d else None


def _percentile(values: List[float], p: float) -> float:
    s = sorted(values)
    return round(s[max(0, math.ceil(p * len(s)) - 1)], 2)


def _latency(values: List[float]) -> Dict[str, float]:
    return (
        {"mean": round(statistics.mean(values), 2), "p50": _percentile(values, 0.5), "p95": _percentile(values, 0.95)}
        if values else {}
    )  # fmt: skip


def _field_acc(rows: List[Dict[str, Any]], key: str = "fields") -> Optional[float]:
    scores = [s for r in rows for _, s in r.get(key, [])]
    return _pct(sum(scores), len(scores))


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if not r["error"]]
    scored = [r for r in ok if r["has_gold"]]
    out: Dict[str, Any] = {"documents": len(rows), "pipeline_failures": len(rows) - len(ok)}
    out["failures"] = [{"id": r["id"], "error": r["error"]} for r in rows if r["error"]]

    out["routing_accuracy_pct"] = _pct(sum(r["routed_ok"] for r in ok), len(rows))
    out["routing_by_modality"] = {
        m: _pct(sum(r["routed_ok"] for r in ok if r["modality"] == m), sum(1 for r in rows if r["modality"] == m))
        for m in sorted({r["modality"] for r in rows})
    }
    out["confusion"] = dict(Counter(f"{r['expected_type']}->{r['predicted_type']}" for r in ok if not r["routed_ok"]))

    out["field_accuracy_pct"] = _field_acc(scored)
    out["field_accuracy_first_attempt_pct"] = _field_acc(scored, "fields_first")
    out["doc_exact_match_pct"] = _pct(sum(r["exact"] for r in scored), len(scored))
    by_field: Dict[str, List[float]] = defaultdict(list)
    for r in scored:
        for name, s in r["fields"]:
            by_field[name].append(s)
    out["field_accuracy_by_field"] = {k: _pct(sum(v), len(v)) for k, v in sorted(by_field.items())}
    out["exact_by_modality"] = {
        m: _pct(sum(r["exact"] for r in scored if r["modality"] == m), sum(1 for r in scored if r["modality"] == m))
        for m in sorted({r["modality"] for r in scored})
    }
    inj = [r for r in scored if "prompt-injection" in r["tags"]]
    out["prompt_injection"] = {"n": len(inj), "exact_pct": _pct(sum(r["exact"] for r in inj), len(inj))}

    # --- repair loop: same run, attempt 1 vs final
    triggered = [r for r in scored if r["attempts"] > 1]
    out["repair"] = {
        "triggered_docs": len(triggered),
        "fixed": sum(1 for r in triggered if r["exact"] and not r["exact_first"]),
        "regressed": sum(1 for r in triggered if r["exact_first"] and not r["exact"]),
        "exact_first_attempt_pct": _pct(sum(r["exact_first"] for r in scored), len(scored)),
        "exact_final_pct": out["doc_exact_match_pct"],
    }

    # --- is the human-review flag useful?
    wrong = [r for r in scored if not r["exact"]]
    flagged = [r for r in scored if r["needs_review"]]
    caught = [r for r in wrong if r["needs_review"]]
    out["review_flag"] = {
        "flagged": len(flagged), "wrong_docs": len(wrong),
        "precision_pct": _pct(len(caught), len(flagged)), "recall_pct": _pct(len(caught), len(wrong)),
    }  # fmt: skip

    out["latency_s"] = {
        m: {
            "total": _latency([r["timings"]["total_s"] for r in ok if r["modality"] == m]),
            "llm_only": _latency(
                [r["timings"]["route_s"] + r["timings"]["extract_s"] for r in ok if r["modality"] == m]
            ),
            "parsing_mean": round(statistics.mean([r["timings"]["parse_s"] for r in ok if r["modality"] == m]), 3),
        }
        for m in sorted({r["modality"] for r in ok})
    }
    return out


# ------------------------------------------------------------------ reporting
def _f(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1f}%"


def render_markdown(summary: Dict[str, Any], meta: Dict[str, Any]) -> str:
    s, rp, rv = summary, summary["repair"], summary["review_flag"]
    lines = [
        f"_Run `{meta['timestamp']}` on {s['documents']} documents "
        f"({meta['dataset']}); text: `{meta['text_models']}`, vision: `{meta['vision_model']}`; "
        f"{meta['llm_calls']} LLM calls; {s['pipeline_failures']} pipeline failure(s)._",
        "",
        "| Metric | Result |",
        "| :-- | --: |",
        f"| Routing accuracy | **{_f(s['routing_accuracy_pct'])}** |",
        f"| Field-level accuracy (final) | **{_f(s['field_accuracy_pct'])}** |",
        f"| Field-level accuracy (attempt 1, no repair) | {_f(s['field_accuracy_first_attempt_pct'])} |",
        f"| Documents with every field correct | {_f(s['doc_exact_match_pct'])} |",
        f"| Repair loop: triggered / fixed / regressed | {rp['triggered_docs']} / {rp['fixed']} / {rp['regressed']} |",
        f"| Review flag precision / recall | {_f(rv['precision_pct'])} / {_f(rv['recall_pct'])} |",
        f"| Prompt-injection docs fully correct | {_f(s['prompt_injection']['exact_pct'])} (n={s['prompt_injection']['n']}) |",
        "",
        "| Modality | Routing | All fields correct | Latency mean / p95 (s, parsing incl.) | LLM-only mean (s) |",
        "| :-- | --: | --: | --: | --: |",
    ]
    for m, lat in s["latency_s"].items():
        t = lat["total"]
        lines.append(
            f"| {m} | {_f(s['routing_by_modality'].get(m))} | {_f(s['exact_by_modality'].get(m))} | "
            f"{t.get('mean', 'n/a')} / {t.get('p95', 'n/a')} | {lat['llm_only'].get('mean', 'n/a')} |"
        )
    lines += ["", "| Field | Accuracy |", "| :-- | --: |"]
    lines += [f"| `{k}` | {_f(v)} |" for k, v in s["field_accuracy_by_field"].items()]
    if s["confusion"]:
        lines += ["", "Routing errors: " + ", ".join(f"`{k}` x{v}" for k, v in s["confusion"].items())]
    if s["failures"]:
        lines += ["", "Pipeline failures: " + "; ".join(f"`{f['id']}` ({f['error'][:80]})" for f in s["failures"])]
    return "\n".join(lines)


def update_readme(readme: Path, block: str) -> None:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.S)
    if not pattern.search(text):
        raise SystemExit(f"{readme} has no {README_START} ... {README_END} markers")
    readme.write_text(pattern.sub(lambda _: f"{README_START}\n{block}\n{README_END}", text), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="evaluation/dataset")
    ap.add_argument("--modality", choices=["all", "text", "image"], default="all")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-repairs", type=int)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between documents (free-tier rate limits)")
    ap.add_argument("--out", default="evaluation/results")
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args(argv)

    from docidp.llm import LangChainBackend

    settings = Settings.from_env()
    if args.max_repairs is not None:
        settings = settings.with_overrides(max_repairs=args.max_repairs)
    backend = LangChainBackend(settings)
    dataset = Path(args.dataset)
    rows = run_evaluation(dataset, backend, settings, args.modality, args.limit, progress=True, delay=args.delay)
    summary = summarize(rows)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    meta = {
        "timestamp": stamp, "dataset": f"synthetic, seed {json.loads((dataset / 'labels.json').read_text())['seed']}",
        "text_models": f"{settings.text_provider}: {settings.router_model} / {settings.extractor_model}", "vision_model": f"{settings.vision_provider}:{settings.vision_model}",
        "llm_calls": backend.calls, "max_repairs": settings.max_repairs,
    }  # fmt: skip
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval-{stamp}.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2, default=str), encoding="utf-8"
    )
    block = render_markdown(summary, meta)
    (out / "latest.md").write_text(block + "\n", encoding="utf-8")
    print("\n" + block)
    if args.update_readme:
        update_readme(Path("README.md"), block)
        print("\nREADME.md updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
