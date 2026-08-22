"""LLM-powered analysis layer for Smart Inbox AI."""

from .analyzer import EmailAnalyzer
from .language import LanguageResult, detect_language, validate_language
from .provider import LLMProvider, OpenAICompatibleProvider, MockLLMProvider
from .schemas import EmailAnalysis

__all__ = [
    "EmailAnalysis",
    "EmailAnalyzer",
    "LanguageResult",
    "LLMProvider",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "detect_language",
    "validate_language",
]
