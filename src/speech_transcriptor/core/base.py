"""
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import noisereduce as nr
import numpy as np
import pandas as pd
from pydub import AudioSegment
from scipy.io import wavfile


@dataclass
class TranscriptionSegment:
    """Сегмент диаризации с информацией о говорящем.
    
    Args:
        start (float): старт в миллисекундах.
        end (float): 
        speaker (str): Может быть его не надо определять
    """

    start: float
    end: float
    text: str
    speaker: str | None = None


class BaseSpeechRecognizer(ABC):
    """
    """

    def __init__(self, device: Literal["cuda", "cpu"] = "cuda") -> None:
        """Инициализация распознавателя речи.

        Args:
            device (Literal["cuda", "cpu"]): Устройство для вычислений. Defaults to "cuda".
        """
        self.logger = logging.getLogger(self.__class__.__name__)

    def _convert_input_file(
        self,
        audio_path: str | Path,
        output_path: str | Path,
        target_sample_rate: int = 16000,
    ) -> Path:
        """Конвертация аудио в WAV с нужной частотой дискретизации и моно каналом.

        Args:
            audio_path (str | Path): Путь к исходному аудиофайлу.
            output_path (str | Path | None): Путь для сохранения конвертированного файла.
                Если None, создается рядом с исходным. Defaults to None.
            target_sample_rate (int): Целевая частота дискретизации. Defaults to 16000.

        Returns:
            Path: Путь к файлу, конвертированному в WAV формат.
        """

        audio_path = Path(audio_path)
        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_converted.wav"
        else:
            output_path = Path(output_path)

        self.logger.debug("Конвертация аудио: %s -> %s", audio_path, output_path)
        audio_segment: AudioSegment = AudioSegment.from_file(audio_path)
        audio_segment = audio_segment.set_frame_rate(target_sample_rate)
        audio_segment: AudioSegment = audio_segment.set_channels(1)

        audio_segment.export(output_path, format="wav")
        self.logger.debug("Конвертация завершена. Результат сохранён в файл: %s", output_path)
        return output_path

    def _reduce_noise_in_audio(
        self,
        wav_path: str | Path,
        output_path: str | Path | None = None,
        noise_sample_sec: float = 0.5,
    ) -> Path:
        """Удаление шума из аудио файла WAV.

        Args:
            wav_path (str | Path): Путь к WAV аудиофайлу.
            output_path (str | Path | None): Путь для сохранения результата.
                Если None, создается рядом с исходным. Defaults to None.
            noise_sample_sec (float): Длительность отрывка из начала файла,
                используемая для образца шума.

        Returns:
            Path: Путь к файлу c подавленными шумами.
        """
        wav_path = Path(wav_path)
        if output_path is None:
            output_path = wav_path.parent / f"{wav_path.stem}_denoised.wav"
        else:
            output_path = Path(output_path)

        self.logger.debug("Применение шумоподавления: %s -> %s", wav_path, output_path)
        rate, data = wavfile.read(str(wav_path))
        # Преобразование данных к float32 в диапазоне [-1, 1]
        if data.dtype == np.int16:
            audio_data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio_data = data.astype(np.float32) / 2147483648.0
        elif np.issubdtype(data.dtype, np.floating):
            audio_data = data.astype(np.float32)
        else:
            raise ValueError(f"Неизвестный формат данных аудио: {data.dtype}")

        # Получение образца шума из начала записи, но не длиннее файла
        noise_sample_len = min(int(noise_sample_sec * rate), len(audio_data) // 10)
        noise_sample = audio_data[:noise_sample_len]

        # Применение шумоподавления
        reduced_noise = nr.reduce_noise(
            y=audio_data,
            sr=rate,
            y_noise=noise_sample,
            stationary=False,
        )

        # Конвертация обратно к int16 для сохранения
        denoised_int16 = np.clip(reduced_noise * 32768.0, -32768, 32767).astype(np.int16)
        wavfile.write(str(output_path), rate, denoised_int16)

        self.logger.info("Шумоподавление завершено: %s", output_path)
        return output_path

    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> list[TranscriptionSegment]:
        """
        """
        pass

    @abstractmethod
    def diarize(
        self,
        audio_path: str | Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> pd.DataFrame:
        """
        """
        pass

    @abstractmethod
    def __call__(
        self,
        audio_path: str | Path,
        language: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> list[TranscriptionSegment]:
        """
        """
        pass
