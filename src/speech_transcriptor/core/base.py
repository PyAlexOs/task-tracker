import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class TranscriptionSegment:
    """Сегмент распознанного текста с временными метками."""

    start: float
    end: float
    text: str
    speaker: str | None = None

    def __repr__(self) -> str:
        return 


@dataclass
class DiarizationSegment:
    """Сегмент диаризации с информацией о говорящем."""

    start: float
    end: float
    speaker: str


class BaseSpeechRecognizer(ABC):
    """Абстрактный базовый класс для транскрибатора речи со встроенной диаризацией.

    Определяет интерфейс для работы с моделями распознавания речи
    и диаризации говорящих. Поддерживает как API-based, так и
    локальные реализации моделей.
    """

    def __init__(self, device: str | None = None) -> None:
        """Инициализация распознавателя речи.

        Args:
            device (Literal['cuda', 'cpu']): Устройство для вычислений ('cuda', 'cpu', если None,
                используется cuda). Defaults to None.
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.logger.info("Инициализация распознавателя на устройстве: %s", self.device)

    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> list[TranscriptionSegment]:
        """Распознавание речи из аудиофайла.

        Args:
            audio_path (str | Path): Путь к аудиофайлу.
            language (str | None): Язык аудио (ISO 639-1 код, например 'ru', 'en').
                Если None, язык определяется автоматически.

        Returns:
            Список сегментов с распознанным текстом и временными метками.
        """
        pass

    @abstractmethod
    def diarize(
        self,
        audio_path: str | Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[DiarizationSegment]:
        """Диаризация говорящих в аудиофайле.

        Args:
            audio_path (str | Path): Путь к аудиофайлу.
            min_speakers (int | None): Минимальное количество говорящих.
            max_speakers: (int | None): Максимальное количество говорящих.

        Returns:
            Список сегментов с информацией о говорящих.
        """
        pass

    @abstractmethod
    def transcribe_with_diarization(
        self,
        audio_path: str | Path,
        language: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[TranscriptionSegment]:
        """Распознавание речи с диаризацией говорящих.

        Args:
            audio_path: Путь к аудиофайлу.
            language: Язык аудио (ISO 639-1 код).
            min_speakers: Минимальное количество говорящих.
            max_speakers: Максимальное количество говорящих.

        Returns:
            Список сегментов с распознанным текстом и информацией о говорящих.
        """
        pass
