"""LLM-powered analysis layer for Smart Inbox AI."""

from .analyzer import EmailAnalyzer
from .provider import LLMProvider, OpenAICompatibleProvider, MockLLMProvider
from .schemas import EmailAnalysis

__all__ = [
    "EmailAnalysis",
    "EmailAnalyzer",
    "LLMProvider",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
]
