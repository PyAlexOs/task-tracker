from typing import Literal

from openai import OpenAI

from src.text_summarizer.core.base import BaseLLM, SummarizationType


class OpenAILLM(BaseLLM):
    """Реализация для работы через OpenAI API (или совместимые API)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        max_tokens: int = 2048,
    ):
        """Конструктор класса.

        Args:
            api_key (str): API ключ.
            base_url (str): Базовый URL API (для совместимых API).
            model_name (str): Название модели.
            max_tokens (int): Максимальное количество токенов. Defaults to 2048.
        """
        super().__init__()

        self._api_key = api_key
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=base_url,
        )

        self.model_name = model_name
        self.max_tokens = max_tokens

    @property
    def summary_type2prompt(self) -> dict[SummarizationType | Literal["RAG"], str]:
        """Маппинг типов суммаризации на системные промпты."""
        # TODO вынести промпты в pydantic модели
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

    def make_rich_prompt(self, prompt: dict[str, str]) -> list[dict[str, str]]:
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
        """Генерация ответа через OpenAI API.

        Args:
            prompt (list[dict[str, str]]): Список сообщений в формате OpenAI API
            temperature (float): Параметр температуры для управления случайностью генерации
                (0.0 - детерминированный, 1.0+ - более творческий).
            max_tokens (int | None): Максимальная длина ответа в токенах. Defaults to None.

        Returns:
            str: Результат генерации.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

        except Exception:  # pylint: disable=broad-exception-caught
            self.logger.exception("Ошибка при генерации ответа.")
            raise
