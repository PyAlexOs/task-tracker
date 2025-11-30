"""Интерфейс базовой обёртки LLM и запросов на суммаризацию, RAG."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal, Self


class SummarizationType(Enum):
    """Типы саммаризации.

    Перечисление доступных режимов суммаризации текста.
    """

    TEXT = auto()
    """Стандартная суммаризация текста, как одного целого."""

    BY_SPEAKERS = auto()
    """Суммаризация текста по спикерам (или ведущим беседу в текстовом виде пользователям).
    По сути является набором запросов на суммаризацию - для каждого спикера отдельно.
    """

    BY_TOPICS = auto()
    """Суммаризация текста по темам (задчам). Для такого вида суммаризации
    необходим также список задач обсуждаемой канбан-доски.
    """


@dataclass
class SummarizationRequest:
    """Запрос на саммаризацию."""

    summarization_type: SummarizationType
    """Тип саммаризации."""

    text: str
    """Исходный текст для саммаризации."""

    speakers: list[str] | None = None
    """Список спикеров для режима DIALOGUE_BY_SPEAKERS."""

    topics: list[str] | None = None
    """Список тем для режимов BY_TOPICS."""

    max_length: int | None = None
    """Максимальная длина саммаризации в токенах или символах."""

    temperature: float = 0.5
    """Параметр температуры для управления случайностью генерации (0.0 - детерминированный, 1.0+ - более творческий)"""

    @classmethod
    def common_summary(
        cls,
        summarization_type: Literal[SummarizationType.TEXT],
        text: str,
        temperature: float = 0.5,
        max_length: int | None = None,
    ) -> Self:
        """Создаёт запрос на обычную суммаризацию текста.

        Args:
            summarization_type (Literal[SummarizationType.TEXT]): Тип саммаризации.
            text (str): Исходный текст для саммаризации.
            temperature (float): Параметр температуры для управления случайностью генерации
                (0.0 - детерминированный, 1.0+ - более творческий). Defaults to 0.5.
            max_length (int | None, optional): Максимальная длина саммаризации в токенах или символах.
                Defaults to None.

        Returns:
            SummarizationRequest: Запрос на суммаризацию текста.
        """

        return cls(
            summarization_type=summarization_type,
            text=text,
            temperature=temperature,
            max_length=max_length,
        )

    @classmethod
    def by_speakers(
        cls,
        summarization_type: Literal[SummarizationType.BY_SPEAKERS],
        text: str,
        speakers: list[str],
        temperature: float = 0.5,
        max_length: int | None = None,
    ) -> Self:
        """Создаёт запрос на суммаризацию текста по спикерам (или ведущим беседу
        в текстовом виде пользователям).

        Args:
            summarization_type (Literal[SummarizationType.BY_SPEAKERS]): Тип саммаризации.
            text (str): Исходный текст для саммаризации.
            speakers (list[str]): Список спикеров.
            temperature (float): Параметр температуры для управления случайностью генерации
                (0.0 - детерминированный, 1.0+ - более творческий). Defaults to 0.5.
            max_length (int | None, optional): Максимальная длина саммаризации в токенах или символах.
                Defaults to None.

        Returns:
            SummarizationRequest: Запрос на суммаризацию текста по спикерам.
        """

        return cls(
            summarization_type=summarization_type,
            text=text,
            speakers=speakers,
            temperature=temperature,
            max_length=max_length,
        )

    @classmethod
    def by_topics(
        cls,
        summarization_type: Literal[SummarizationType.BY_TOPICS],
        text: str,
        topics: list[str],
        temperature: float = 0.5,
        max_length: int | None = None,
    ) -> Self:
        """Создаёт запрос на суммаризацию текста по темам.

        Args:
            summarization_type (Literal[SummarizationType.BY_TOPICS]): Тип саммаризации.
            text (str): Исходный текст для саммаризации.
            topics (list[str]): Темы для саммаризации.
            temperature (float): Параметр температуры для управления случайностью генерации
                (0.0 - детерминированный, 1.0+ - более творческий). Defaults to 0.5.
            max_length (int | None, optional): Максимальная длина саммаризации в токенах или символах.
                Defaults to None.

        Returns:
            SummarizationRequest: Запрос на суммаризацию текста по темам.
        """

        return cls(
            summarization_type=summarization_type,
            text=text,
            topics=topics,
            temperature=temperature,
            max_length=max_length,
        )


class BaseLLM(ABC):
    """Абстрактный базовый класс для работы с LLM.
    Определяет интерфейс для саммаризации и RAG.
    """

    def __init__(self, **kwargs: Any):
        """Конструктор класса."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def summary_type2prompt(self) -> dict[SummarizationType | Literal["RAG"], str]:
        """Абстрактный маппинг вида {тип суммаризации: промпт для LLM}.
        Должен быть реализован в подклассах.
        """

    @abstractmethod
    def _make_rich_prompt(self, prompt: dict[str, str]) -> Any:
        """Преобразует готовые промпты для LLM из словарей таким образом,
        чтобы используемая модель поддерживала ввод.

        Args:
            prompt (dict[str, str]): Словарь частей промпта,
                из которых можно составить цельный промпт.

        Returns:
            Any: Промпт строкой или словарем, в завивсимости
                от возможностей API или бэкэнда используемой модели.
        """

    @abstractmethod
    def _generate(
        self,
        prompt: Any,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """Генерации ответа LLM.

        Args:
            prompt (Any): Полный промпт для LLM.
            max_tokens (int | None, optional): Максимальная длина ответа в токенах. Defaults to None.
            temperature (float): Параметр температуры для управления случайностью генерации
                (0.0 - детерминированный, 1.0+ - более творческий).

        Returns:
            str: Результат генерации.
        """

    def summarize(self, request: SummarizationRequest) -> str | dict[str, str]:
        """Универсальный метод саммаризации.

        Args:
            request (SummarizationRequest): Объект запроса с параметрами саммаризации.

        Returns:
            str | dict[str, str]: Результат суммаризации. При `SummarizationRequest.summarization_type`
                - `SummarizationType.TEXT` - сгенерированный ответ (строка).
                - `SummarizationType.BY_SPEAKERS` - набор сгенерированных ответов для каждого спикера,
                    словарь вида {спикер: результат суммаризации}.
                - `SummarizationTypeBY_TOPICS` - набор сгенерированных ответов для каждой темы,
                    словарь вида {тема: результат суммаризации}.
        """

        self.logger.debug(
            "Запрос на суммаризацию типа '%s', длина текста: %d символов, temperature=%.2f",
            request.summarization_type.name,
            len(request.text),
            request.temperature,
        )

        prompt_parts: dict[str, str] = {}
        system_prompt = self.summary_type2prompt.get(request.summarization_type)

        if system_prompt is None:
            self.logger.error(
                "Маппинг типа запроса '%s' к промптам не определен",
                request.summarization_type.name,
            )
            raise RuntimeError("Маппинга типов запроса к промптам не был определен.")
        prompt_parts["system"] = system_prompt

        match request.summarization_type:
            case SummarizationType.TEXT:
                # В таком случае никаких дополнительных действий не нужно
                pass

            case SummarizationType.BY_SPEAKERS:
                if not request.speakers:
                    raise ValueError("Не указаны спикеры для суммаризации.")
                prompt_parts["user"] = f"Список спикеров: {request.speakers}.\n\n"

            case SummarizationType.BY_TOPICS:
                if not request.topics:
                    raise ValueError("Не указаны темы для суммаризации.")
                prompt_parts["user"] = f"Список тем: {request.topics}.\n\n"

        user_prompt = prompt_parts.get("user", "")
        prompt_parts["user"] = user_prompt + request.text

        full_prompt = self._make_rich_prompt(prompt_parts)

        start = time.perf_counter_ns()
        result = self._generate(
            full_prompt,
            request.temperature,
            request.max_length,
        )
        end = time.perf_counter_ns()

        self.logger.debug(
            "Суммаризация завершена за %.3f секунд, результат: %d символов",
            (end - start) / 1e9,
            len(result) if isinstance(result, str) else sum(len(v) for v in result.values()),
        )
        return result

    def generate_with_rag(
        self,
        query: str,
        retrieved_contexts: list[str],
        temperature: float,
        max_length: int | None = None,
    ) -> str:
        """Генерация ответа на основе найденных контекстов (RAG).

        Args:
            query (str): Вопрос пользователя.
            retrieved_contexts (list[str]):Список извлеченных релевантных контекстов из базы знаний.
            temperature (float): Параметр температуры для управления случайностью генерации
                (0.0 - детерминированный, 1.0+ - более творческий).
            max_length (int | None): Максимальная длина генерируемого ответа в токенах.
                Defaults to None.

        Returns:
            str: Сгенерированный ответ на основе предоставленного контекста.
        """

        self.logger.debug(
            "Запрос в RAG: %d документов, длина запроса: %d символов, temperature=%.2f",
            len(retrieved_contexts),
            len(query),
            temperature,
        )

        prompt_parts: dict[str, str] = {}
        system_prompt = self.summary_type2prompt.get("RAG")

        if system_prompt is None:
            raise RuntimeError(
                "При переопределении маппинга типов запроса к промптам не был определен промпт для RAG."
            )
        prompt_parts["system"] = system_prompt

        context_text = "\n".join(
            [f"[Документ {i + 1}]:\n{ctx}" for i, ctx in enumerate(retrieved_contexts)]
        )
        prompt_parts["user"] = f"КОНТЕКСТ:\n{context_text}\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{query}"

        full_prompt = self._make_rich_prompt(prompt_parts)

        start = time.perf_counter_ns()
        answer = self._generate(
            full_prompt,
            temperature,
            max_length,
        )
        end = time.perf_counter_ns()

        self.logger.debug(
            "RAG ответ сгенерирован за %.3f секунд, длина ответа: %d символов",
            (end - start) / 1e9,
            len(answer),
        )
        return answer
