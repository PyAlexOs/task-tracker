"""Модуль, содержащий реализации локально поднятой LLM и обёртки для хождения в LLM по API."""

from .api_summarizer import OpenAILLM
from .local_summarizer import LocalLLM

__all__ = [
    "OpenAILLM",
    "LocalLLM",
]
