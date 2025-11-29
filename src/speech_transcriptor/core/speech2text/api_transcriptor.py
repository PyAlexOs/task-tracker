from pathlib import Path

from src.speech_transcriptor.core.base import (
    BaseSpeechRecognizer,
    TranscriptionSegment,
)


class APISpeechRecognizer(BaseSpeechRecognizer):
    """Реализация распознавателя речи через API.

    Использует внешние API для распознавания речи и диаризации,
    не загружая модели локально.
    """

    def __init__(
        self,
        api_key: str,
        api_endpoint: str,
        device: str | None = None,
    ) -> None:
        """
        Инициализация API распознавателя.

        Args:
            api_key: Ключ API для аутентификации.
            api_endpoint: URL эндпоинта API.
            device: Устройство (не используется для API, но сохраняется для совместимости).
        """
        super().__init__(device)
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.logger.info("Инициализация API распознавателя: %s", api_endpoint)

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> list[TranscriptionSegment]:
        """Распознавание речи через API."""

        self.logger.info(f"Отправка запроса на распознавание через API: {audio_path}")

        # response = requests.post(
        #     f"{self.api_endpoint}/transcribe",
        #     headers={"Authorization": f"Bearer {self.api_key}"},
        #     files={"audio": open(audio_path, "rb")},
        #     json={"language": language},
        # )
        # segments = [TranscriptionSegment(**seg) for seg in response.json()["segments"]]

        self.logger.warning("API метод transcribe требует реализации")
        raise NotImplementedError("API метод требует реализации с конкретным API")

    def diarize(
        self,
        audio_path: str | Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list:
        """Диаризация через API."""

        self.logger.info(f"Отправка запроса на диаризацию через API: {audio_path}")

        # Здесь должна быть реализация вызова API
        self.logger.warning("API метод diarize требует реализации")
        raise NotImplementedError("API метод требует реализации с конкретным API")

    def transcribe_with_diarization(
        self,
        audio_path: str | Path,
        language: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[TranscriptionSegment]:
        """Распознавание с диаризацией через API."""

        self.logger.info(f"Отправка запроса на распознавание с диаризацией через API: {audio_path}")

        # Здесь должна быть реализация вызова API
        self.logger.warning("API метод transcribe_with_diarization требует реализации")
        raise NotImplementedError("API метод требует реализации с конкретным API")
