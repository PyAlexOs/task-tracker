"""Модуль, содержащий функциональность для работы с LLM: составление саммари и RAG."""

from .base import BaseLLM, SummarizationRequest, SummarizationType
from .text2summary import LocalLLM, OpenAILLM

__all__ = [
    "BaseLLM",
    "SummarizationRequest",
    "SummarizationType",
    "LocalLLM",
    "OpenAILLM",
]
