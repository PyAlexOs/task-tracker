"""Базовая функциональность для конвертации аудиофайлов в необходимый формат,
удаления шумов, транскрибации и диаризации аудио.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, overload

import noisereduce as nr
import numpy as np
import pandas as pd
import pyannote.audio
import torch
import whisper
from numpy.typing import NDArray
from pydub import AudioSegment
from pydub.effects import normalize
from scipy.io import wavfile

type ModelName = Literal[
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "large-v3-turbo",
    "turbo",
]


@dataclass
class TranscriptionWord:
    start: float
    end: float
    text: str
    speaker: str | None = None
    probs: float | None = None


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    words: list[TranscriptionWord] = field(default_factory=list)


class SpeechTranscriptor:
    """Транскрибатор речи с диаризацией спикеров."""

    def __init__(
        self,
        whisper_model_name: ModelName | Path,
        diarizer_model_name: str,
        hf_token: str,
        device: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """Конструктор класса.

        Args:
            model_name (ModelName | Path): Название модели Whisper для загрузки или путь к локальным файлам модели.
            diarizer_model_name (str): Название модели диаризации для загрузки.
            hf_token (str): HuggingFace токен для доступа к моделям диаризации.
            device (Literal["cuda", "cpu"]): Устройство для вычислений. Defaults to "cuda".
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        self.whisper_model_name = whisper_model_name
        self.device = device

        self.logger.debug("Инициализация транскрибатора на устройстве: %s", self.device)
        self.logger.debug("Загрузка модели OpenAI Whisper: %s", self.whisper_model_name)
        self.whisper_model = whisper.load_model(self.whisper_model_name, device=device)

        self.logger.debug("Загрузка модели диаризации pyannote.audio")
        # Для загрузки весов из "небезопасного источника"
        with torch.serialization.safe_globals(
            [
                torch.torch_version.TorchVersion,
                pyannote.audio.core.task.Specifications,
                pyannote.audio.core.task.Problem,
                pyannote.audio.core.task.Resolution,
            ]
        ):
            # NOTE для изменения модели, необходимо выдать API ключу доступ до этой модели в hf
            self.diarization_pipeline = pyannote.audio.Pipeline.from_pretrained(
                diarizer_model_name,
                token=hf_token,
            )
        self.diarization_pipeline.to(torch.device(self.device))

    @overload
    def _convert_input_file(
        self,
        audio_path: str | Path,
        output_path: str | Path | None = None,
        target_sample_rate: int = 16000,
        save_to_disk: Literal[True] = True,
    ) -> Path:
        """Конвертация аудио в WAV с нужной частотой дискретизации и моно каналом.

        Args:
            audio_path (str | Path): Путь к исходному аудиофайлу.
            output_path (str | Path | None): Путь для сохранения конвертированного файла.
                Если None, создается рядом с исходным. Defaults to None.
            target_sample_rate (int): Целевая частота дискретизации. Defaults to 16000.
            save_to_disk (bool): Возвращать конвертированный формат аудиозаписи (если False)
                или сохранять аудиозапись на диск и возвращать путь к сохраненному файлу. Defaults to False.

        Returns:
            Path: Путь к файлу, конвертированному в WAV формат.
        """

    @overload
    def _convert_input_file(
        self,
        audio_path: str | Path,
        target_sample_rate: int = 16000,
        save_to_disk: Literal[False] = False,
    ) -> tuple[int, NDArray[Any]]:
        """Конвертация аудио в WAV с нужной частотой дискретизации и моно каналом.

        Args:
            audio_path (str | Path): Путь к исходному аудиофайлу.
            target_sample_rate (int): Целевая частота дискретизации. Defaults to 16000.
            save_to_disk (bool): Возвращать конвертированный формат аудиозаписи (если False)
                или сохранять аудиозапись на диск и возвращать путь к сохраненному файлу. Defaults to False.

        Returns:
           tuple[int, NDArray[Any]]: Частота дискретизации и аудиозапись в конвертированном формате.
        """

    def _convert_input_file(
        self,
        audio_path: str | Path,
        output_path: str | Path | None = None,
        target_sample_rate: int = 16000,
        save_to_disk: bool = False,
    ) -> Path | tuple[int, NDArray[Any]]:

        audio_path = Path(audio_path)
        self.logger.debug(
            "Конвертация аудио: %s (target_sr=%d, save_to_disk=%s)",
            audio_path,
            target_sample_rate,
            save_to_disk,
        )

        audio_segment: AudioSegment = AudioSegment.from_file(audio_path)
        audio_segment = normalize(audio_segment)

        audio_segment = audio_segment.set_frame_rate(target_sample_rate)
        audio_segment = audio_segment.set_channels(1)

        if save_to_disk:
            if output_path is None:
                output_path = audio_path.parent / f"{audio_path.stem}_converted.wav"
            else:
                output_path = Path(output_path)

            audio_segment.export(output_path, format="wav")
            self.logger.debug(
                "Конвертация завершена. Результат сохранён в файл: %s", output_path
            )
            return output_path

        # В память: pydub хранит в sample_width байт на сэмпл
        samples = np.array(audio_segment.get_array_of_samples())

        sample_width = audio_segment.sample_width
        if sample_width == 1:  # 8-bit unsigned
            audio_data = ((samples.astype(np.float32) / 128.0) - 1.0) * 32768.0
            audio_data = audio_data.astype(np.int16)
        elif sample_width == 2:  # 16-bit signed
            audio_data = samples.astype(np.int16)
        elif sample_width == 4:  # 32-bit signed
            audio_data = (samples.astype(np.float64) / 2147483648.0 * 32768.0).astype(np.int16)
        else:
            raise ValueError(f"Unsupported sample_width: {sample_width}")

        self.logger.debug(
            "Конвертация завершена. Результат возвращён в память: len=%d",
            len(audio_data),
        )
        return target_sample_rate, audio_data
    
    def _get_noise_sample(self, audio_data: NDArray[Any], rate: int, total_duration: float = 2.0) -> NDArray[Any]:
        """Берет шумовые sample из нескольких частей файла."""
        n_samples = len(audio_data)
        noise_samples = []
        
        # Берем из начала (первые 0.5 сек)
        start_samples = min(int(0.5 * rate), n_samples // 10)
        if start_samples > 0:
            noise_samples.append(audio_data[:start_samples])
        
        # Берем из середины (если файл достаточно длинный)
        if n_samples > 10 * rate:  # Если больше 10 секунд
            mid_start = n_samples // 2 - int(0.5 * rate)
            mid_end = n_samples // 2 + int(0.5 * rate)
            if mid_end < n_samples:
                noise_samples.append(audio_data[mid_start:mid_end])
        
        # Берем из конца (последние 0.5 сек)
        end_samples = min(int(0.5 * rate), n_samples // 10)
        if end_samples > 0:
            noise_samples.append(audio_data[-end_samples:])
        
        if noise_samples:
            return np.concatenate(noise_samples)
        else:
            return audio_data[:min(int(1.0 * rate), n_samples)]

    @overload
    def _reduce_noise_in_audio(
        self,
        wav_path: str | Path,
        output_path: str | Path | None = None,
        noise_sample_sec: float = 0.5,
        save_to_disk: Literal[True] = True,
    ) -> Path:
        """Удаление шума из аудио файла WAV.

        Args:
            wav_path (str | Path): Путь к WAV аудиофайлу.
            output_path (str | Path | None): Путь для сохранения результата.
                Если None, создается рядом с исходным. Defaults to None.
            noise_sample_sec (float): Длительность отрывка из начала файла,
                используемая для образца шума. Defaults to 0.5.
            save_to_disk (bool): Возвращать конвертированный формат аудиозаписи (если False)
                или сохранять аудиозапись на диск и возвращать путь к сохраненному файлу. Defaults to False.

        Returns:
            Path: Путь к файлу c подавленными шумами.
        """

    @overload
    def _reduce_noise_in_audio(
        self,
        wav_path: str | Path,
        noise_sample_sec: float = 0.5,
        save_to_disk: Literal[False] = False,
    ) -> tuple[int, NDArray[Any]]:
        """Удаление шума из аудио файла WAV.

        Args:
            wav_path (str | Path): Путь к WAV аудиофайлу.
            noise_sample_sec (float): Длительность отрывка из начала файла,
                используемая для образца шума. Defaults to 0.5.
            save_to_disk (bool): Возвращать конвертированный формат аудиозаписи (если False)
                или сохранять аудиозапись на диск и возвращать путь к сохраненному файлу. Defaults to False.

        Returns:
            tuple[int, NDArray[Any]]: Представление записи с подавленными шумами.
        """

    @overload
    def _reduce_noise_in_audio(
        self,
        wav_array: tuple[int, NDArray[Any]],
        output_path: str | Path | None = None,
        noise_sample_sec: float = 0.5,
        save_to_disk: Literal[True] = True,
    ) -> Path:
        """Удаление шума из аудио файла WAV.

        Args:
            wav_array (tuple[int, NDArray[Any]]): Частота дискретизации аудио и представление в виде массива.
            output_path (str | Path | None): Путь для сохранения результата.
                Если None, создается рядом с исходным. Defaults to None.
            noise_sample_sec (float): Длительность отрывка из начала файла,
                используемая для образца шума. Defaults to 0.5.
            save_to_disk (bool): Возвращать конвертированный формат аудиозаписи (если False)
                или сохранять аудиозапись на диск и возвращать путь к сохраненному файлу. Defaults to False.

        Returns:
            Path: Путь к файлу c подавленными шумами.
        """

    @overload
    def _reduce_noise_in_audio(
        self,
        wav_array: tuple[int, NDArray[Any]],
        noise_sample_sec: float = 0.5,
        save_to_disk: Literal[False] = False,
    ) -> tuple[int, NDArray[Any]]:
        """Удаление шума из аудио файла WAV.

        Args:
            wav_array (tuple[int, NDArray[Any]]): Частота дискретизации аудио и представление в виде массива.
            noise_sample_sec (float): Длительность отрывка из начала файла,
                используемая для образца шума. Defaults to 0.5.
            save_to_disk (bool): Возвращать конвертированный формат аудиозаписи (если False)
                или сохранять аудиозапись на диск и возвращать путь к сохраненному файлу. Defaults to False.

        Returns:
            tuple[int, NDArray[Any]]: Представление записи с подавленными шумами.
        """

    def _reduce_noise_in_audio(
        self,
        wav_path: str | Path | None = None,
        wav_array: tuple[int, NDArray[Any]] | None = None,
        output_path: str | Path | None = None,
        noise_sample_sec: float = 0.5,
        save_to_disk: bool = True,
    ) -> Path | tuple[int, NDArray[Any]]:
        
        if not wav_path and not wav_array:
            raise RuntimeError(
                "Не передано ни пути к файлу .wav, ни представления звука в виде массива."
            )

        # В приоритете получение из памяти
        elif wav_array:
            rate, data = wav_array

        else:
            wav_path = Path(wav_path)
            self.logger.debug(
                "Применение шумоподавления: %s (noise_sample_sec=%.3f, save_to_disk=%s)",
                wav_path,
                noise_sample_sec,
                save_to_disk,
            )
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
        
        # Для моно: (N,) -> оставляем как есть
        # Для стерео: (N, C) -> берем только первый канал
        if audio_data.ndim == 2:
            self.logger.debug("Многоканальное аудио, извлекается первый канал")
            audio_data = audio_data[:, 0]
        
        if audio_data.ndim != 1:
            raise ValueError(f"Неподдерживаемая форма аудио: {audio_data.shape}")
        
        n_samples = len(audio_data)
        if n_samples == 0:
            raise ValueError("Пустой WAV-файл, нечего обрабатывать")

        # Берем максимум между указанным временем и минимум 1 секундой
        min_noise_samples = max(int(noise_sample_sec * rate), rate)  # минимум 1 секунда
        # Но не больше 20% файла (вместо 10%)
        max_noise_samples = n_samples // 5
        noise_sample_len = min(min_noise_samples, max_noise_samples)

        # Fallback для очень коротких файлов
        if noise_sample_len <= 0:
            noise_sample_len = min(n_samples, rate // 2)  # 0.5 секунды
        noise_sample = self._get_noise_sample(audio_data, rate)

        reduced = nr.reduce_noise(
            y=audio_data,
            sr=rate,
            y_noise=noise_sample,
            stationary=False,
            prop_decrease=0.8,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
        )

        if reduced.ndim != 1:
            raise ValueError(f"Неожиданная форма после обработки: {reduced.shape}")

        denoised_int16 = np.clip(reduced * 32768.0, -32768, 32767).astype(np.int16)

        if save_to_disk:
            if output_path is None:
                if wav_path is None:
                    raise RuntimeError(
                        "При save_to_disk=True и использовании wav_array необходимо передать output_path."
                    )
                output_path = wav_path.parent / f"{wav_path.stem}_denoised.wav"
            else:
                output_path = Path(output_path)

            wavfile.write(str(output_path), rate, denoised_int16)
            self.logger.debug("Шумоподавление завершено: %s", output_path)
            return output_path

        self.logger.debug(
            "Шумоподавление завершено. Результат возвращён в память: len=%d",
            len(denoised_int16),
        )
        return rate, denoised_int16

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> list[TranscriptionSegment]:
        """Транскрибирует аудиофайл в список текстовых сегментов.

        Args:
            audio_path (str | Path): Путь к аудиофайлу, который необходимо транскрибировать.
            language (str | None): Необязательный BCP-47 код языка речи
                (например, "ru-RU" или "en-US"). Если не указан, язык может
                определяться автоматически в зависимости от используемого бэкенда.

        Returns:
            list[TranscriptionSegment]: Список сегментов транскрипции с временными
                метками и распознанным текстом.
        """

        self.logger.debug("Начало распознавания: %s", audio_path)
        kwargs: dict[str, Any] = {}
        if language is not None:
            kwargs["language"] = language

        result = self.whisper_model.transcribe(
            str(audio_path),
            word_timestamps=True,
            **kwargs,
        )

        segments: list[TranscriptionSegment] = []
        for seg in result.get("segments", []):
            words_list = [
                TranscriptionWord(
                    start=w.get("start", 0.0),
                    end=w.get("end", 0.0),
                    text=w.get("word", "").strip(),
                    # speaker и probs могут быть добавлены позже
                    speaker=w.get("speaker"),
                    probs=w.get("probability"),
                ) for w in seg.get("words", [])
            ]

            segments.append(
                TranscriptionSegment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", "").strip(),
                    words=words_list,
                )
            )

        self.logger.debug("Распознано %d сегментов", len(segments))
        return segments

    def diarize(
        self,
        audio_path: str | Path,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> pd.DataFrame:
        """Выполняет диаризацию аудио и возвращает разметку по спикерам.

        Args:
            audio_path (str | Path): Путь к аудиофайлу, для которого нужно
                выполнить диаризацию.
            min_speakers (int | None): Минимальное предполагаемое количество
                спикеров в записи. Если None, минимальное число спикеров
                выбирается автоматически.
            max_speakers (int | None): Максимальное предполагаемое количество
                спикеров в записи. Если None, максимальное число спикеров
                выбирается автоматически.

        Returns:
            pd.DataFrame: Таблица с результатами диаризации, содержащая как
                минимум столбцы с временными метками и идентификаторами спикеров.
        """

        if self.diarization_pipeline is None:
            self.logger.error("Модель диаризации не загружена")
            raise ValueError("HuggingFace токен требуется для диаризации")

        self.logger.debug("Начало диаризации: %s", audio_path)

        diarization_kwargs: dict[str, Any] = {}
        if min_speakers is not None:
            diarization_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarization_kwargs["max_speakers"] = max_speakers

        diarization = self.diarization_pipeline(
            str(audio_path),
            **diarization_kwargs,
        )
        annotation = diarization.exclusive_speaker_diarization

        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                }
            )

        self.logger.debug("Обнаружено %d сегментов диаризации", len(segments))
        return pd.DataFrame(segments)
    
    def assign_word_speakers(
        self,
        diarization_df: pd.DataFrame,
        asr_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Назначает каждому слову в результате ASR идентификатор спикера по данным диаризации.

        Args:
            diarization_df (pd.DataFrame): Результаты диаризации с временными
                метками и идентификаторами спикеров.
            asr_result (dict[str, Any]): Результат распознавания речи, содержащий
                расшифровку и временные метки слов или сегментов.

        Returns:
            dict[str, Any]: Обновлённый результат ASR, в котором словам или
            сегментам назначены идентификаторы спикеров.
        """

        if "segments" not in asr_result:
            return asr_result

        for seg in asr_result["segments"]:
            words = seg.get("words") or []
            for w in words:
                w_start = w.get("start")
                w_end = w.get("end")
                if w_start is None or w_end is None:
                    continue

                mask = (diarization_df["start"] < w_end) & (diarization_df["end"] > w_start)
                if not mask.any():
                    continue

                overlaps: list[tuple[float, str]] = []
                for _, row in diarization_df[mask].iterrows():
                    overlap = min(w_end, row["end"]) - max(w_start, row["start"])
                    if overlap > 0:
                        overlaps.append((overlap, row["speaker"]))

                if overlaps:
                    _, best_speaker = max(overlaps, key=lambda x: x[0])
                    w["speaker"] = best_speaker

        return asr_result

    def __call__(
        self,
        audio_path: str | Path,
        language: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        target_sample_rate: int = 16000,
        noise_sample_sec: float = 0.5,
    ) -> dict[str, Any]:
        """Проводит распознавание речи и диаризацию для заданного аудиофайла.

        Args:
            audio_path (str | Path): Путь к аудиофайлу, который нужно обработать.
            language (str | None): Необязательный BCP-47 код языка речи
                (например, "ru-RU" или "en-US"). Если не указан, язык может
                определяться автоматически в зависимости от бэкенда.
            min_speakers (int | None): Минимальное предполагаемое количество
                спикеров в записи. Если None, значение подбирается автоматически.
            max_speakers (int | None): Максимальное предполагаемое количество
                спикеров в записи. Если None, значение подбирается автоматически.
            target_sample_rate (int): Целевая частота дискретизации для конвертации.
            noise_sample_sec (float): Длина отрезка для оценки шума при шумоподавлении.

        Returns:
            dict[str, Any]: Словарь с результатами обработки, содержащий как
                минимум ключи с транскрипцией и разметкой спикеров.
        """

        self.logger.debug("Запущена обработка файла: %s", audio_path)
        audio_path = Path(audio_path)

        self.logger.debug("Конвертация файла в формат .wav")
        sr, audio_data = self._convert_input_file(
            audio_path=audio_path,
            target_sample_rate=target_sample_rate,
            save_to_disk=False,
        )

        self.logger.debug("Удаление шумов из аудио.")
        clear_wav_filepath = self._reduce_noise_in_audio(
            wav_array=(sr, audio_data),
            output_path=audio_path.parent / f"{audio_path.stem}_denoised.wav",
            noise_sample_sec=noise_sample_sec,
            save_to_disk=True,
        )

        self.logger.debug("Транскрибация аудио.")
        asr_kwargs: dict[str, Any] = {}
        if language is not None:
            asr_kwargs["language"] = language

        asr_result: dict[str, Any] = self.whisper_model.transcribe(
            str(clear_wav_filepath),
            word_timestamps=True,
            **asr_kwargs,
        )
        print(asr_result)

        self.logger.debug("Диаризация аудио.")
        diarization_df = self.diarize(
            audio_path=clear_wav_filepath,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        asr_result = self.assign_word_speakers(diarization_df, asr_result)
        self.logger.debug(
            "Пайплайн завершён: %s (segments=%d)",
            audio_path,
            len(asr_result.get("segments", [])),
        )
        return asr_result
