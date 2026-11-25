"""LLM backends. The pipeline depends only on the small LLMBackend protocol, so tests and
evaluations can swap in scripted backends without touching the network."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Mapping, Optional, Protocol

from pydantic import BaseModel

from .config import Settings
from .errors import ConfigError, ExtractionError
from .parsers import Document
from .prompts import ROUTER_SYSTEM, build_user_text, extractor_system
from .schemas import SCHEMA_REGISTRY, RouteDecision


class LLMBackend(Protocol):
    calls: int

    def route(self, doc: Document) -> RouteDecision: ...

    def extract(
        self, doc: Document, doc_type: str, instruction: str = "", feedback: Optional[str] = None
    ) -> BaseModel: ...


class LangChainBackend:
    """Groq or OpenAI for text, Groq or OpenAI for images. Only the keys actually needed are required.

    With the defaults (TEXT_PROVIDER=groq, VISION_PROVIDER=groq) a single free Groq key runs everything.
    """

    def __init__(self, settings: Settings, env: Optional[Mapping[str, str]] = None):
        self.settings = settings
        self._env = os.environ if env is None else env
        self._models: Dict[str, Any] = {}
        self._chains: Dict[str, Any] = {}
        self.calls = 0

    # ---- model construction (lazy, so importing the package never needs keys or network)
    def _key(self, name: str) -> str:
        key = self._env.get(name)
        if not key:
            raise ConfigError(f"{name} is not set. Add it to your environment or .env file.")
        return key

    def _chat_model(self, role: str, is_image: bool):
        s = self.settings
        provider = s.vision_provider if is_image else s.text_provider
        model = s.vision_model if is_image else (s.router_model if role == "router" else s.extractor_model)
        cache_key = f"{provider}:{model}"
        if cache_key not in self._models:
            common = dict(
                model=model, temperature=s.temperature, timeout=s.llm_timeout_s, max_retries=s.llm_max_retries
            )
            if provider == "groq":
                from langchain_groq import ChatGroq

                self._models[cache_key] = ChatGroq(api_key=self._key("GROQ_API_KEY"), **common)
            else:
                from langchain_openai import ChatOpenAI

                self._models[cache_key] = ChatOpenAI(api_key=self._key("OPENAI_API_KEY"), **common)
        return cache_key, self._models[cache_key]

    def _structured(self, role: str, is_image: bool, schema: type):
        cache_key, model = self._chat_model(role, is_image)
        chain_key = f"{cache_key}:{schema.__name__}"
        if chain_key not in self._chains:
            # function_calling is the most widely supported structured-output mode (Groq + OpenAI)
            self._chains[chain_key] = model.with_structured_output(schema, method="function_calling")
        return self._chains[chain_key]

    def _invoke(self, chain, messages, what: str, model_var: str = "the model setting"):
        last: Optional[Exception] = None
        for attempt in range(2):  # SDK already retries network/429; this retries malformed tool output
            try:
                self.calls += 1
                result = chain.invoke(messages)
                if result is None:
                    raise ValueError("model returned no structured output")
                return result
            except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
                last = exc
                if "model_not_found" in str(exc) or "does not exist" in str(exc):
                    # Providers retire preview models: retrying is pointless, tell the user how to fix it.
                    raise ExtractionError(
                        f"{what} failed: the model is unavailable (retired or no access). "
                        f"Set {model_var} in your .env to a current model "
                        f"(Groq: https://console.groq.com/docs/models). Provider said: {exc}"
                    ) from exc
                time.sleep(0.5 * (attempt + 1))
        raise ExtractionError(f"{what} failed: {last}") from last

    @staticmethod
    def _messages(system: str, user_text: str, doc: Document, vision_provider: str = "groq"):
        from langchain_core.messages import HumanMessage, SystemMessage

        if doc.is_image:
            image_url = {"url": f"data:image/jpeg;base64,{doc.content}"}
            if vision_provider == "openai":  # `detail` is an OpenAI-only extension; Groq documents only `url`
                image_url["detail"] = "high"
            human = HumanMessage(
                content=[{"type": "text", "text": user_text}, {"type": "image_url", "image_url": image_url}]
            )
        else:
            human = HumanMessage(content=user_text)
        return [SystemMessage(content=system), human]

    # ---- public API
    def route(self, doc: Document) -> RouteDecision:
        sample = doc.content[: self.settings.router_sample_chars] if not doc.is_image else None
        user = build_user_text(is_image=doc.is_image, content=sample)
        chain = self._structured("router", doc.is_image, RouteDecision)
        messages = self._messages(ROUTER_SYSTEM, user, doc, self.settings.vision_provider)
        return self._invoke(chain, messages, "routing", "VISION_MODEL" if doc.is_image else "ROUTER_MODEL")

    def extract(self, doc: Document, doc_type: str, instruction: str = "", feedback: Optional[str] = None) -> BaseModel:
        schema = SCHEMA_REGISTRY[doc_type]
        user = build_user_text(
            is_image=doc.is_image,
            content=None if doc.is_image else doc.content,
            instruction=instruction,
            feedback=feedback,
        )
        chain = self._structured("extractor", doc.is_image, schema)
        messages = self._messages(extractor_system(doc_type), user, doc, self.settings.vision_provider)
        model_var = "VISION_MODEL" if doc.is_image else "EXTRACTOR_MODEL"
        return self._invoke(chain, messages, f"{doc_type} extraction", model_var)