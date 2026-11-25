"""docidp: schema-constrained document extraction with validation-driven self-repair."""

__version__ = "2.0.0"

from .config import Settings
from .parsers import Document, load_document
from .pipeline import DocumentPipeline, ExtractionResult

__all__ = ["Settings", "Document", "load_document", "DocumentPipeline", "ExtractionResult"]
