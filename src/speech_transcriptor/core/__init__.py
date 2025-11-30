"""Модуль, содержащий базовую функциональность для конвертации аудиофайлов в необходимый формат,
удаления шумов, транскрибации и диаризации аудио.
"""

from .base import (
    ModelName,
    SpeechTranscriptor,
    TranscriptionSegment,
    TranscriptionWord,
)

__all__ = [
    "ModelName",
    "SpeechTranscriptor",
    "TranscriptionSegment",
    "TranscriptionWord",
]
