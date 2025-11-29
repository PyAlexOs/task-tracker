from pathlib import Path

import pandas as pd
import torch
import whisperx
from pyannote.audio import Pipeline

from src.speech_transcriptor.core.base import (
    BaseSpeechRecognizer,
    TranscriptionSegment,
)


class LocalSpeechRecognizer(BaseSpeechRecognizer):
    """
    Реализация распознавателя речи с локальной загрузкой моделей.

    Использует WhisperX для распознавания речи и pyannote.audio
    для диаризации говорящих.
    """

    def __init__(
        self,
        model_name: str = "small",
        hf_token: str | None = None,
        device: str | None = None,
        compute_type: str = "float16",
    ) -> None:
        """Инициализация локального распознавателя.

        Args:
            model_name: Название модели Whisper ('tiny', 'base', 'small', 'medium', 'large-v2').
            hf_token: HuggingFace токен для доступа к моделям диаризации.
            device: Устройство для вычислений.
            compute_type: Тип вычислений ('float16', 'int8').
        """
        self.device = device
        self.logger.debug("Инициализация распознавателя на устройстве: %s", self.device)

        self.model_name = model_name
        self.compute_type = compute_type
        self.hf_token = hf_token

        self.logger.info("Загрузка модели Whisper: %s", model_name)
        self.whisper_model = whisperx.load_model(
            model_name,
            device=self.device,
            compute_type=compute_type,
        )

        if hf_token:
            self.logger.info("Загрузка модели диаризации pyannote.audio")
            # NOTE для изщменения модели, необходимо выдать API ключу доступ до этой модели в hf
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            self.diarization_pipeline.to(torch.device(self.device))

        else:
            self.diarization_pipeline = None
            self.logger.warning("HuggingFace токен не предоставлен, диаризация недоступна")

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> list[TranscriptionSegment]:
        """Распознавание речи из аудиофайла."""
        self.logger.debug("Начало распознавания: %s", audio_path)
        audio = whisperx.load_audio(str(audio_path))

        result = self.whisper_model.transcribe(
            audio,
            language=language,
            batch_size=16,
        )

        segments = [
            TranscriptionSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            )
            for seg in result["segments"]
        ]

        self.logger.info("Распознано %d сегментов", len(segments))
        return segments

    def diarize(
        self,
        audio_path: str | Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> pd.DataFrame:
        """Диаризация говорящих в аудиофайле."""

        if self.diarization_pipeline is None:
            self.logger.error("Модель диаризации не загружена")
            raise ValueError("HuggingFace токен требуется для диаризации")

        self.logger.debug("Начало диаризации: %s", audio_path)

        diarization_kwargs = {}
        if min_speakers is not None:
            diarization_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_kwargs["max_speakers"] = max_speakers

        diarization = self.diarization_pipeline(
            str(audio_path),
            **diarization_kwargs,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        self.logger.info(f"Обнаружено {len(segments)} сегментов диаризации")
        return pd.DataFrame(segments)

    def transcribe_with_diarization(
        self,
        audio_path: str | Path,
        language: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> dict:
        """Распознавание речи с диаризацией говорящих."""

        if self.diarization_pipeline is None:
            self.logger.error("Модель диаризации не загружена")
            raise ValueError("HuggingFace токен требуется для диаризации")

        self.logger.info("Начало распознавания с диаризацией: %s", audio_path)
        audio = whisperx.load_audio(str(audio_path))

        result = self.whisper_model.transcribe(
            audio,
            language=language,
            batch_size=16,
        )

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device=self.device,
        )

        diarization_df = self.diarize(audio_path, min_speakers, max_speakers)

        # Присвоение говорящих к словам
        return whisperx.assign_word_speakers(diarization_df, result)
