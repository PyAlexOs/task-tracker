from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class SummarizationType(Enum):
    """Типы саммаризации."""

    TEXT = "text"
    DIALOGUE_BY_SPEAKERS = "dialogue_by_speakers"
    BY_TOPICS = "by_topics"
    UNTOUCHED_TOPICS = "untouched_topics"


@dataclass
class SummarizationRequest:
    """Запрос на саммаризацию."""

    text: str
    summarization_type: SummarizationType
    speakers: list[str] | None = None  # Для DIALOGUE_BY_SPEAKERS
    topics:list[str] | None = None  # Для BY_TOPICS и UNTOUCHED_TOPICS
    max_length: int | None = None


@dataclass
class RAGRequest:
    """Запрос для RAG генерации."""

    query: str
    retrieved_contexts: list[str]
    max_length: int | None = None
    temperature: float = 0.7


class BaseLLM(ABC):
    """
    Абстрактный базовый класс для работы с LLM.
    Определяет интерфейс для саммаризации и RAG.
    """

    @abstractmethod
    def summarize_text(self, text: str, max_length: int | None = None) -> str:
        """
        Саммаризация обычного текста.
        
        Args:
            text: Текст для саммаризации
            max_length: Максимальная длина саммари
            
        Returns:
            Саммаризованный текст
        """
        pass

    @abstractmethod
    def summarize_dialogue_by_speakers(
        self, 
        dialogue: str, 
        speakers: list[str],
        max_length: int | None = None
    ) -> dict[str, str]:
        """
        Саммаризация полилога с разбивкой по спикерам.
        
        Args:
            dialogue: Текст диалога/полилога
            speakers: Список имен спикеров
            max_length: Максимальная длина саммари для каждого спикера
            
        Returns:
            Словарь {спикер: саммари_его_высказываний}
        """
        pass

    @abstractmethod
    def summarize_by_topics(
        self, 
        text: str, 
        topics: list[str],
        max_length: int | None = None
    ) -> dict[str, str]:
        """
        Выделение и саммаризация по указанным темам.
        
        Args:
            text: Текст для анализа
            topics: Список тем для выделения
            max_length: Максимальная длина саммари для каждой темы
            
        Returns:
            Словарь {тема: саммари_по_этой_теме}
        """
        pass

    @abstractmethod
    def find_untouched_topics(
        self, 
        text: str, 
        expected_topics: list[str]
    ) -> list[str]:
        """
        Определение тем из списка, которые НЕ были затронуты в тексте.
        
        Args:
            text: Текст для анализа
            expected_topics: Список ожидаемых тем
            
        Returns:
            Список незатронутых тем
        """
        pass

    @abstractmethod
    def generate_with_rag(
        self, 
        query: str, 
        retrieved_contexts: list[str],
        max_length: int | None = None,
        temperature: float = 0.7
    ) -> str:
        """
        Генерация ответа на основе найденных контекстов (RAG).
        
        Args:
            query: Вопрос пользователя
            retrieved_contexts: Список найденных релевантных текстов
            max_length: Максимальная длина ответа
            temperature: Температура генерации
            
        Returns:
            Сгенерированный ответ
        """
        pass

    def summarize(self, request: SummarizationRequest) -> Any:
        """
        Универсальный метод саммаризации.
        
        Args:
            request: Объект запроса с параметрами саммаризации
            
        Returns:
            Результат саммаризации (str или Dict)
        """
        match request.summarization_type:
            case SummarizationType.TEXT:
                return self.summarize_text(request.text, request.max_length)
        
            case SummarizationType.DIALOGUE_BY_SPEAKERS:
                if not request.speakers:
                    raise ValueError("Speakers list required for DIALOGUE_BY_SPEAKERS")
                return self.summarize_dialogue_by_speakers(
                    request.text, 
                    request.speakers, 
                    request.max_length
                )
            
            case SummarizationType.BY_TOPICS:
                if not request.topics:
                    raise ValueError("Topics list required for BY_TOPICS")
                return self.summarize_by_topics(
                    request.text, 
                    request.topics, 
                    request.max_length
                )
            
            case SummarizationType.UNTOUCHED_TOPICS:
                if not request.topics:
                    raise ValueError("Topics list required for UNTOUCHED_TOPICS")
                return self.find_untouched_topics(request.text, request.topics)