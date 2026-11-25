# Architecture and design decisions

## Pipeline

1. **Parse** (`parsers.py`): bytes → `Document(kind="text"|"image")`.
   - PDF → text via `pypdf` (layout mode keeps table columns aligned) or `marker` (markdown, OCR). `auto` uses marker only if installed.
     Page and size limits are enforced *before* parsing; a PDF without a text layer raises `ScannedPDFError` with a remedy.
   - Image → upright (EXIF), white background (alpha flattened), RGB, ≤2048 px JPEG, base64.
2. **Route** (`llm.py`): a larger model sees the first 3,000 characters (or the image) and returns `RouteDecision(type, confidence, reasoning)`.
3. **Extract**: the specialist for that type returns a Pydantic object via function-calling. Fields are required-but-nullable.
4. **Normalise + verify** (`validators.py`): ISO dates, ISO-4217 currency codes; then checks.
   *Errors* (trigger repair): missing/non-positive total, negative tax, tax > total, `subtotal + tax ≠ total` (tolerance max(0.05, 0.5 %)), no records.
   *Warnings* (shown, no repair): missing vendor/number/date, non-ISO currency, receipt arithmetic (tips and discounts are normal).
5. **Repair** (`pipeline.py`): on errors, re-extract with the failing checks in the prompt, up to `MAX_REPAIRS`. The prompt tells the model to
   change a value only if it *misread* it, never to make numbers add up. The attempt with the fewest errors is kept (ties → earliest).
6. **Flag**: `needs_review` if errors remain, routing confidence < threshold, or the text was truncated.
7. **Export** (`export.py`): Excel (Summary / Invoices / Receipts / General / Validation sheets) and JSON.

## Decisions and trade-offs

| Decision | Why | Cost |
| :-- | :-- | :-- |
| Router + specialists instead of one prompt | Small focused schemas are easier to enforce and evaluate per type | Extra call; misrouting is a new failure mode (measured) |
| Deterministic verifier, not "LLM as judge" | Arithmetic is exactly checkable; cheap, reproducible, explainable | Only catches errors that violate a rule (a wrong vendor name passes; the eval reports this as review-flag *recall*) |
| Best-of-attempts, ties keep the first | A repair loop must never degrade the result | Slightly more code |
| Nullable fields | v1's required strings forced hallucinated values | Consumers must handle `None` |
| Backend `Protocol` | Pipeline and eval are testable offline with scripted/oracle backends | One more abstraction |
| `pypdf` default, `marker` optional | `marker` pulls PyTorch and GBs of weights; most PDFs have a text layer | Layout quality on complex tables is lower without marker |
| Groq as default for text *and* vision, OpenAI optional | One free key runs the whole project; provider is a setting, not a code change | Groq vision models are previews and may rotate; free-tier token caps need throttling (`--delay`) |
| Image payload guard (≤3.5 MB base64) | Groq rejects base64 images above 4 MB; compressing quality, then resolution, avoids a runtime 413 | Very large scans lose some resolution |
| `detail` hint only for OpenAI | Groq documents only `url` in `image_url`; unknown keys risk a 400 | Provider-specific branch in one function |
| Sequential eval | Honest latency numbers, no rate-limit noise | Slower runs |
| Attempt-1 vs final measured in one run | Quantifies the repair loop's value without a second API pass | Only applies to documents where repair triggered |

## Threat model

| Risk | Mitigation |
| :-- | :-- |
| Prompt injection inside a document | Content fenced in `<document>` tags (closing tag neutralised), system prompts declare it untrusted, user guidance length-capped, injection traps in the benchmark with a dedicated metric. *Mitigation, not a guarantee.* |
| Formula injection via Excel | Cells starting with `=` are written as text (tested by round-trip) |
| Malicious / huge uploads | Size, page-count and pixel limits (Pillow decompression-bomb guard); corrupt files → `ParsingError` |
| Shell injection | `marker` is invoked with an argument list (no shell) on a fixed filename in a temp dir, with a timeout |
| Secret leakage | Keys read only from the environment; `.env` is git-ignored |
| Data exposure to providers | Documented; provider is configurable |

## Testing strategy

No test touches the network. `ScriptedBackend` drives exact repair-loop scenarios; `OracleBackend` replays the ground truth so the whole evaluation harness
(including real PDF parsing and image normalisation over all 43 generated files) can be verified to score 100 % when the "model" is perfect, and to detect
injected faults. A fake `marker_single` executable on `PATH` covers the subprocess path (success, failure, timeout, no output).
