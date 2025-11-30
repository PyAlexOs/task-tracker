"""Реализация для локальной LLM с использованием vLLM."""

from pathlib import Path
from typing import Any, Literal

from vllm import LLM, SamplingParams

from ..base import BaseLLM, SummarizationType


class LocalLLM(BaseLLM):
    """Реализация для локальной LLM с использованием vLLM."""

    def __init__(
        self,
        model_path: str | Path,
        device: Literal["cuda", "cpu"] = "cuda",
        max_tokens: int = 2048,
        **kwargs: Any,
    ):
        """Инициализация локальной модели через vLLM.

        Args:
            model_path (str): Путь к модели или идентификатор на HuggingFace.
            device (Literal["cuda", "cpu"]): Устройство для вычислений. Defaults to "cuda".
            max_tokens (int): Максимальное количество токенов. Defaults to 2048.
            **kwargs (Any): Дополнительные параметры для vLLM (tensor_parallel_size, gpu_memory_utilization и др.)
        """

        super().__init__()
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens

        self._client = LLM(
            model=str(self.model_path),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )

    @property
    def summary_type2prompt(self) -> dict[SummarizationType | Literal["RAG"], str]:
        """Маппинг типов суммаризации на системные промпты."""
        # TODO вынести промпты в pydantic модели + вынести в родительский класс промпты, если они будут в конфигах
        # TODO сделать fallback на запрос отдельно по спикерам, если общий json не валидный
        return {
            SummarizationType.TEXT: (
                "Ты — профессиональный ассистент для суммаризации текста. "
                "Создай краткое и информативное резюме предоставленного текста, "
                "сохраняя ключевые моменты и основные идеи."
            ),
            SummarizationType.BY_SPEAKERS: (
                "Ты — профессиональный ассистент для анализа полилогов. "
                "Для каждого спикера из предоставленного списка создай отдельное резюме "
                "того, что этот спикер сказал или обсудил в тексте. "
                'Верни результат в формате JSON: {"спикер1": "резюме1", "спикер2": "резюме2", ...}'
            ),
            SummarizationType.BY_TOPICS: (
                "Ты — профессиональный ассистент для тематического анализа текста. "
                "Для каждой темы из предоставленного списка извлеки и суммируй "
                "информацию, связанную с этой темой. Если тема не упоминается в тексте, "
                "не добавляй её в результат. Если ты не нашел ни одной темы, верни пустой JSON."
                'Верни результат в формате JSON: {"тема1": "резюме1", "тема2": "резюме2", ...}'
            ),
            "RAG": (
                "Ты — профессиональный ассистент, который отвечает на вопросы пользователя "
                "на основе предоставленного контекста. Используй только информацию из контекста. "
                "Если ответ не может быть найден в контексте, честно сообщи об этом.\n"
                "ВАЖНЫЕ ПРАВИЛА\n:"
                "- Отвечай ТОЛЬКО на основе информации из предоставленных документов\n"
                "- Если информации недостаточно для ответа, явно укажи это\n"
                "- Не используй внешние знания, если они не подтверждены контекстом\n"
                "- Будь точен и конкретен в ответах."
            ),
        }

    def _make_rich_prompt(self, prompt: dict[str, str]) -> list[dict[str, str]]:
        """Преобразует словарь промптов в формат сообщений OpenAI API.

        Args:
            prompt (dict[str, str]): Словарь с ключами "system" и "user".

        Returns:
            list[dict[str, str]]: Список сообщений в формате OpenAI API.
        """
        messages = []

        if "system" in prompt:
            messages.append({"role": "system", "content": prompt["system"]})

        if "user" in prompt:
            messages.append({"role": "user", "content": prompt["user"]})

        return messages

    def _generate(
        self,
        prompt: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None = None,
    ) -> str:
        """Генерация ответа через vLLM.

        Args:
            prompt (list[dict[str, str]]): Список сообщений в формате chat template.
            temperature (float): Параметр температуры для управления случайностью генерации.
            max_tokens (int | None): Максимальная длина ответа в токенах. Defaults to None.

        Returns:
            str: Результат генерации.
        """

        try:
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens or self.max_tokens,
            )

            # vLLM поддерживает метод chat() для работы с chat templates
            outputs = self._client.chat(
                messages=prompt,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            return outputs[0].outputs[0].text.strip()

        except Exception:  # pylint: disable=broad-exception-caught
            self.logger.exception("Ошибка при генерации ответа.")
            raise
