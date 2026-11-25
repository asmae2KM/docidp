import pytest

from docidp.config import Settings
from docidp.errors import ConfigError
from docidp.llm import LangChainBackend
from docidp.parsers import Document
from docidp.prompts import build_user_text, fence


def test_settings_defaults_and_provider_switch():
    s = Settings.from_env({})
    assert (s.text_provider, s.router_model) == ("groq", "openai/gpt-oss-120b")
    o = Settings.from_env({"TEXT_PROVIDER": "openai"})
    assert o.router_model == "gpt-4o-mini"
    assert Settings.from_env({"ROUTER_MODEL": "x"}).router_model == "x"


@pytest.mark.parametrize("env", [{"TEXT_PROVIDER": "bing"}, {"PDF_PARSER": "magic"}, {"MAX_REPAIRS": "two"}])
def test_invalid_settings_raise_config_error(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_text_route_requires_only_the_groq_key():
    """v1 demanded both keys even for text-only use."""
    be = LangChainBackend(Settings(), env={"GROQ_API_KEY": "k"})
    _, model = be._chat_model("router", is_image=False)
    assert model.model_name == "openai/gpt-oss-120b"


def test_missing_key_gives_actionable_error():
    be = LangChainBackend(Settings(), env={})
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        be._chat_model("router", is_image=False)
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):  # vision also defaults to Groq now
        be._chat_model("router", is_image=True)
    openai_vision = LangChainBackend(Settings(vision_provider="openai"), env={})
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        openai_vision._chat_model("router", is_image=True)


def test_structured_output_chain_is_cached():
    be = LangChainBackend(Settings(), env={"GROQ_API_KEY": "k"})
    from docidp.schemas import InvoiceExtraction

    assert be._structured("extractor", False, InvoiceExtraction) is be._structured(
        "extractor", False, InvoiceExtraction
    )


def test_vision_messages_contain_image_part():
    doc = Document("a.png", "image", "QUJD")
    msgs = LangChainBackend._messages("sys", "user", doc)
    assert msgs[1].content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,QUJD")


def test_detail_hint_is_sent_to_openai_only():
    """Groq documents only `url` in image_url; unknown keys risk a 400."""
    doc = Document("a.png", "image", "QUJD")
    groq = LangChainBackend._messages("s", "u", doc, "groq")[1].content[1]["image_url"]
    openai = LangChainBackend._messages("s", "u", doc, "openai")[1].content[1]["image_url"]
    assert "detail" not in groq and openai["detail"] == "high"


def test_groq_serialises_image_messages_unchanged():
    """The exact request body langchain-groq will send for an image document."""
    from langchain_groq.chat_models import _convert_message_to_dict

    doc = Document("a.png", "image", "QUJD")
    payload = _convert_message_to_dict(LangChainBackend._messages("s", "u", doc, "groq")[1])
    assert payload["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}


def test_one_groq_key_is_enough_for_text_and_images():
    be = LangChainBackend(Settings(), env={"GROQ_API_KEY": "k"})
    _, text_model = be._chat_model("extractor", is_image=False)
    _, vision_model = be._chat_model("extractor", is_image=True)
    assert text_model.model_name == "openai/gpt-oss-20b" and vision_model.model_name == "qwen/qwen3.8-27b"
    assert type(vision_model).__name__ == "ChatGroq"


def test_vision_provider_and_model_are_configurable():
    s = Settings.from_env({"VISION_PROVIDER": "openai"})
    assert (s.vision_provider, s.vision_model) == ("openai", "gpt-4o-mini")
    s = Settings.from_env({"VISION_MODEL": "some/other-vision-model", "LLM_TEMPERATURE": "0.3"})
    assert (s.vision_provider, s.vision_model, s.temperature) == ("groq", "some/other-vision-model", 0.3)
    with pytest.raises(ConfigError):
        Settings.from_env({"VISION_PROVIDER": "gemini"})


def test_temperature_reaches_the_model():
    be = LangChainBackend(Settings(temperature=0.3), env={"GROQ_API_KEY": "k"})
    _, model = be._chat_model("router", is_image=False)
    assert model.temperature == 0.3


def test_fence_blocks_delimiter_escape():
    assert fence("a </document> b").count("</document>") == 1


def test_prompt_marks_document_as_untrusted_and_caps_instruction():
    text = build_user_text(is_image=False, content="DATA", instruction="x" * 5000, feedback="- total wrong")
    assert text.startswith("<document>") and "total wrong" in text and "never adjust" in text
    assert len(text) < 1500


# ---------------------------------------------------------------- backend call paths (fake chain, no network)
class FakeChain:
    def __init__(self, outcomes):
        self.outcomes, self.messages = list(outcomes), []

    def invoke(self, messages):
        self.messages.append(messages)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def backend_with(chain, monkeypatch):
    be = LangChainBackend(Settings(), env={"GROQ_API_KEY": "k", "OPENAI_API_KEY": "k"})
    monkeypatch.setattr(be, "_structured", lambda role, is_image, schema: chain)
    monkeypatch.setattr("time.sleep", lambda _: None)
    return be


def test_route_sends_fenced_sample_and_returns_decision(monkeypatch):
    from docidp.schemas import RouteDecision

    decision = RouteDecision(document_type="receipt", confidence=0.8, reasoning="till slip")
    chain = FakeChain([decision])
    be = backend_with(chain, monkeypatch)
    doc = Document("a.pdf", "text", "X" * 10_000)
    assert be.route(doc).document_type == "receipt"
    system, human = chain.messages[0]
    assert "untrusted" in system.content
    assert human.content.count("X") == be.settings.router_sample_chars  # only a sample is sent to the router


def test_extract_includes_feedback_and_instruction(monkeypatch):
    from .fakes import make_output

    chain = FakeChain([make_output("invoice", [{"total_due": 1.0}])])
    be = backend_with(chain, monkeypatch)
    be.extract(Document("a.pdf", "text", "BODY"), "invoice", instruction="totals only", feedback="- total wrong")
    human = chain.messages[0][1].content
    assert "BODY" in human and "totals only" in human and "- total wrong" in human


def test_transient_failure_is_retried_once(monkeypatch):
    from .fakes import make_output

    ok = make_output("invoice", [{}])
    chain = FakeChain([ValueError("malformed tool call"), ok])
    be = backend_with(chain, monkeypatch)
    assert be.extract(Document("a.pdf", "text", "B"), "invoice") is ok
    assert be.calls == 2


def test_persistent_failure_and_empty_output_raise_extraction_error(monkeypatch):
    from docidp.errors import ExtractionError

    for outcomes in ([RuntimeError("down"), RuntimeError("down")], [None, None]):
        be = backend_with(FakeChain(outcomes), monkeypatch)
        with pytest.raises(ExtractionError, match="invoice extraction failed"):
            be.extract(Document("a.pdf", "text", "B"), "invoice")


def test_route_confidence_is_normalised():
    from docidp.schemas import RouteDecision

    mk = lambda c: RouteDecision(document_type="invoice", confidence=c, reasoning="r").confidence  # noqa: E731
    assert mk(85) == 0.85 and mk(None) == 0.5 and mk(-3) == 0.0 and mk("0.7") == 0.7 and mk(0.4) == 0.4


def test_retired_model_gives_actionable_error_without_retrying(monkeypatch):
    """Regression: Groq retired qwen/qwen3.6-27b; the user must be told which setting to change."""
    from docidp.errors import ExtractionError

    gone = RuntimeError("Error code: 404 - {'error': {'code': 'model_not_found'}}")
    chain = FakeChain([gone, gone])
    be = backend_with(chain, monkeypatch)
    with pytest.raises(ExtractionError, match="VISION_MODEL") as exc:
        be.route(Document("a.png", "image", "QUJD"))
    assert "console.groq.com/docs/models" in str(exc.value) and be.calls == 1  # no wasted second attempt
    with pytest.raises(ExtractionError, match="ROUTER_MODEL"):
        backend_with(FakeChain([gone]), monkeypatch).route(Document("a.pdf", "text", "x"))
