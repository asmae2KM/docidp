"""Exception hierarchy. Everything the library raises on purpose derives from DocIDPError."""


class DocIDPError(Exception):
    """Base class for all expected errors."""


class ConfigError(DocIDPError):
    """Missing API key, unknown provider, invalid setting."""


class UnsupportedFileError(DocIDPError):
    """File extension is not a supported PDF or image type."""


class FileTooLargeError(DocIDPError):
    """Upload exceeds the configured size limit."""


class ParsingError(DocIDPError):
    """A PDF or image could not be read."""


class ScannedPDFError(ParsingError):
    """PDF has no extractable text layer (probably a scan)."""


class ExtractionError(DocIDPError):
    """The LLM call failed or returned nothing usable after retries."""
