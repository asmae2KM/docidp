"""Runtime settings. Read from the environment *at call time*, never at import time."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Mapping, Optional

from .errors import ConfigError

_PROVIDER_DEFAULTS = {
    # Groq vision models are "preview" and rotate: check https://console.groq.com/docs/vision and set VISION_MODEL.
    "groq": {"router": "openai/gpt-oss-120b", "extractor": "openai/gpt-oss-20b", "vision": "qwen/qwen3.8-27b"},
    "openai": {"router": "gpt-4o-mini", "extractor": "gpt-4o-mini", "vision": "gpt-4o-mini"},
}
_PDF_PARSERS = {"auto", "pypdf", "marker"}


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    text_provider: str = "groq"
    router_model: str = "openai/gpt-oss-120b"
    extractor_model: str = "openai/gpt-oss-20b"
    vision_provider: str = "groq"  # groq | openai (independent of text_provider)
    vision_model: str = "qwen/qwen3.8-27b"
    temperature: float = 0.0
    pdf_parser: str = "auto"  # auto | pypdf | marker
    max_repairs: int = 2  # extra extraction attempts after a failed validation
    min_router_confidence: float = 0.6
    max_upload_mb: int = 15
    max_pdf_pages: int = 30
    max_text_chars: int = 30_000
    router_sample_chars: int = 3_000
    marker_timeout_s: int = 300
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Settings":
        env = os.environ if env is None else env
        provider = env.get("TEXT_PROVIDER", "groq").strip().lower()
        if provider not in _PROVIDER_DEFAULTS:
            raise ConfigError(f"TEXT_PROVIDER must be one of {sorted(_PROVIDER_DEFAULTS)}, got {provider!r}")
        vision_provider = env.get("VISION_PROVIDER", "groq").strip().lower()
        if vision_provider not in _PROVIDER_DEFAULTS:
            raise ConfigError(f"VISION_PROVIDER must be one of {sorted(_PROVIDER_DEFAULTS)}, got {vision_provider!r}")
        pdf_parser = env.get("PDF_PARSER", "auto").strip().lower()
        if pdf_parser not in _PDF_PARSERS:
            raise ConfigError(f"PDF_PARSER must be one of {sorted(_PDF_PARSERS)}, got {pdf_parser!r}")
        d = _PROVIDER_DEFAULTS[provider]
        return cls(
            text_provider=provider,
            router_model=env.get("ROUTER_MODEL") or d["router"],
            extractor_model=env.get("EXTRACTOR_MODEL") or d["extractor"],
            vision_provider=vision_provider,
            vision_model=env.get("VISION_MODEL") or _PROVIDER_DEFAULTS[vision_provider]["vision"],
            temperature=_float(env, "LLM_TEMPERATURE", 0.0),
            pdf_parser=pdf_parser,
            max_repairs=max(0, _int(env, "MAX_REPAIRS", 2)),
            min_router_confidence=_float(env, "MIN_ROUTER_CONFIDENCE", 0.6),
            max_upload_mb=_int(env, "MAX_UPLOAD_MB", 15),
            max_pdf_pages=_int(env, "MAX_PDF_PAGES", 30),
            max_text_chars=_int(env, "MAX_TEXT_CHARS", 30_000),
            marker_timeout_s=_int(env, "MARKER_TIMEOUT_S", 300),
            llm_timeout_s=_float(env, "LLM_TIMEOUT_S", 60.0),
            llm_max_retries=_int(env, "LLM_MAX_RETRIES", 2),
        )

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)
