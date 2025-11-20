import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Класс для представления документа."""
    
    content: str
    metadata: dict[str, Any] | None = None
    
    def __post_init__(self) -> None:
        """Инициализация метаданных по умолчанию."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SummarizationResult:
    """Класс для результата суммаризации."""
    
    summary: str
    method: str
    metadata: dict[str, Any] | None = None


class BaseLLMSummarizer(ABC):
    """
    Абстрактный базовый класс для суммаризации текстов и RAG.
    
    Определяет общий интерфейс для работы с LLM моделями,
    независимо от типа развертывания (локально или через API).
    """
    
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """
        Инициализация базового суммаризатора.
        
        Args:
            model_name: Название модели для использования
            **kwargs: Дополнительные параметры конфигурации
        """
        self.model_name = model_name
        self.config = kwargs
        logger.info(f"Инициализация суммаризатора с моделью: {model_name}")
    
    @abstractmethod
    def _generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Базовый метод генерации ответа от LLM.
        
        Args:
            prompt: Промпт для модели
            **kwargs: Дополнительные параметры генерации
            
        Returns:
            Сгенерированный текст
        """
        pass
    
    @abstractmethod
    def _encode_text(self, text: str) -> Any:
        """
        Создание векторного представления текста.
        
        Args:
            text: Текст для кодирования
            
        Returns:
            Векторное представление
        """
        pass
    
    def summarize_extractive(self, text: str, num_sentences: int = 3) -> SummarizationResult:
        """
        Экстрактивная суммаризация: выделение ключевых предложений.
        
        Args:
            text: Исходный текст для суммаризации
            num_sentences: Количество предложений в итоговой суммаризации
            
        Returns:
            Результат суммаризации
        """
        logger.info("Выполнение экстрактивной суммаризации")
        
        prompt = f"""Выдели {num_sentences} наиболее важных предложения из следующего текста.
            Верни только выбранные предложения без изменений.

            Текст: {text}

            Важные предложения:"""
        
        summary = self._generate(prompt, max_tokens=500)
        
        return SummarizationResult(
            summary=summary.strip(),
            method="extractive",
            metadata={"num_sentences": num_sentences}
        )
    
    def summarize_abstractive(self, text: str, max_length: int = 150) -> SummarizationResult:
        """
        Абстрактивная суммаризация: генерация нового текста.
        
        Args:
            text: Исходный текст для суммаризации
            max_length: Максимальная длина суммаризации в словах
            
        Returns:
            Результат суммаризации
        """
        logger.info("Выполнение абстрактивной суммаризации")
        
        prompt = f"""Создай краткую суммаризацию следующего текста (не более {max_length} слов).
Суммаризация должна содержать основные идеи и выводы.

Текст: {text}

Суммаризация:"""
        
        summary = self._generate(prompt, max_tokens=max_length * 2)
        
        return SummarizationResult(
            summary=summary.strip(),
            method="abstractive",
            metadata={"max_length": max_length}
        )
    
    def summarize_multi_document(self, documents: list[Document], max_length: int = 200) -> SummarizationResult:
        """
        Суммаризация нескольких документов.
        
        Args:
            documents: Список документов для суммаризации
            max_length: Максимальная длина суммаризации в словах
            
        Returns:
            Результат суммаризации
        """
        logger.info(f"Выполнение мульти-документной суммаризации ({len(documents)} документов)")
        
        combined_text = "\n\n---\n\n".join([f"Документ {i+1}:\n{doc.content}" 
                                            for i, doc in enumerate(documents)])
        
        prompt = f"""Создай общую суммаризацию следующих документов (не более {max_length} слов).
Объедини ключевые идеи из всех документов.

{combined_text}

Общая суммаризация:"""
        
        summary = self._generate(prompt, max_tokens=max_length * 2)
        
        return SummarizationResult(
            summary=summary.strip(),
            method="multi_document",
            metadata={"num_documents": len(documents), "max_length": max_length}
        )
    
    def rag_query(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 3,
        context_instruction: str | None = None
    ) -> str:
        """
        Ответ на вопрос с использованием RAG (Retrieval-Augmented Generation).
        
        Args:
            query: Вопрос пользователя
            documents: База документов для поиска
            top_k: Количество наиболее релевантных документов
            context_instruction: Дополнительная инструкция для контекста
            
        Returns:
            Ответ на основе найденных документов
        """
        logger.info(f"Выполнение RAG запроса: {query[:50]}...")
        
        # Найти наиболее релевантные документы
        relevant_docs = self._retrieve_relevant_documents(query, documents, top_k)
        
        # Подготовить контекст
        context = "\n\n".join([f"Источник {i+1}:\n{doc.content}" 
                               for i, doc in enumerate(relevant_docs)])
        
        instruction = context_instruction or "Используй только информацию из предоставленных источников."
        
        prompt = f"""Ты AI-ассистент. {instruction}

Контекст:
{context}

Вопрос: {query}

Ответ:"""
        
        answer = self._generate(prompt, max_tokens=500)
        
        logger.info("RAG запрос успешно выполнен")
        return answer.strip()
    
    def _retrieve_relevant_documents(
        self,
        query: str,
        documents: list[Document],
        top_k: int
    ) -> list[Document]:
        """
        Поиск наиболее релевантных документов для запроса.
        
        Args:
            query: Поисковый запрос
            documents: База документов
            top_k: Количество документов для возврата
            
        Returns:
            Список наиболее релевантных документов
        """
        logger.debug(f"Поиск {top_k} релевантных документов")
        
        # Кодирование запроса
        query_embedding = self._encode_text(query)
        
        # Вычисление схожести с документами
        similarities = []
        for doc in documents:
            doc_embedding = self._encode_text(doc.content)
            similarity = self._compute_similarity(query_embedding, doc_embedding)
            similarities.append((doc, similarity))
        
        # Сортировка по релевантности
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in similarities[:top_k]]
    
    @abstractmethod
    def _compute_similarity(self, embedding1: Any, embedding2: Any) -> float:
        """
        Вычисление схожести между двумя векторными представлениями.
        
        Args:
            embedding1: Первое векторное представление
            embedding2: Второе векторное представление
            
        Returns:
            Оценка схожести
        """
        pass


class LocalLLMSummarizer(BaseLLMSummarizer):
    """
    Реализация суммаризатора с использованием локальной LLM модели.
    
    Использует локально развернутую модель (например, через transformers или llama.cpp).
    """
    
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        device: str = "cuda",
        **kwargs: Any
    ) -> None:
        """
        Инициализация локального суммаризатора.
        
        Args:
            model_name: Название модели
            model_path: Путь к файлам модели
            device: Устройство для вычислений ("cuda" или "cpu")
            **kwargs: Дополнительные параметры
        """
        super().__init__(model_name, **kwargs)
        self.model_path = model_path
        self.device = device
        
        # Здесь должна быть инициализация локальной модели
        # Например, с использованием transformers:
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        logger.info(f"Локальная модель загружена на устройство: {device}")
    
    def _generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Генерация ответа с использованием локальной модели.
        
        Args:
            prompt: Промпт для модели
            **kwargs: Параметры генерации (temperature, max_tokens и т.д.)
            
        Returns:
            Сгенерированный текст
        """
        logger.debug("Генерация ответа локальной моделью")
        
        # Пример реализации с transformers:
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=kwargs.get("max_tokens", 512),
        #     temperature=kwargs.get("temperature", 0.7),
        #     do_sample=True
        # )
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Заглушка для примера
        return f"[LOCAL_MODEL_RESPONSE] {prompt[:100]}..."
    
    def _encode_text(self, text: str) -> Any:
        """
        Создание векторного представления с использованием локальной модели эмбеддингов.
        
        Args:
            text: Текст для кодирования
            
        Returns:
            Векторное представление (numpy array или tensor)
        """
        logger.debug("Создание эмбеддинга локальной моделью")
        
        # Пример с sentence-transformers:
        # from sentence_transformers import SentenceTransformer
        # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        # return embedding_model.encode(text)
        
        # Заглушка для примера
        import numpy as np
        return np.random.rand(384)
    
    def _compute_similarity(self, embedding1: Any, embedding2: Any) -> float:
        """
        Вычисление косинусной схожести между векторами.
        
        Args:
            embedding1: Первый вектор
            embedding2: Второй вектор
            
        Returns:
            Косинусная схожесть
        """
        import numpy as np
        
        # Косинусная схожесть
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        return float(dot_product / (norm1 * norm2))


class APILLMSummarizer(BaseLLMSummarizer):
    """
    Реализация суммаризатора с использованием API (OpenAI, Anthropic и т.д.).
    
    Отправляет запросы к внешнему API для генерации и эмбеддингов.
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        api_base_url: str | None = None,
        embedding_model: str = "text-embedding-ada-002",
        **kwargs: Any
    ) -> None:
        """
        Инициализация API суммаризатора.
        
        Args:
            model_name: Название модели (например, "gpt-4", "claude-3-opus")
            api_key: API ключ для аутентификации
            api_base_url: Базовый URL API (опционально)
            embedding_model: Модель для создания эмбеддингов
            **kwargs: Дополнительные параметры
        """
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.embedding_model = embedding_model
        
        # Здесь должна быть инициализация API клиента
        # Например, для OpenAI:
        # from openai import OpenAI
        # self.client = OpenAI(api_key=api_key, base_url=api_base_url)
        
        logger.info(f"API клиент инициализирован для модели: {model_name}")
    
    def _generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Генерация ответа через API.
        
        Args:
            prompt: Промпт для модели
            **kwargs: Параметры генерации (temperature, max_tokens и т.д.)
            
        Returns:
            Сгенерированный текст
        """
        logger.debug("Отправка запроса к API")
        
        # Пример для OpenAI API:
        # response = self.client.chat.completions.create(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=kwargs.get("max_tokens", 512),
        #     temperature=kwargs.get("temperature", 0.7)
        # )
        # return response.choices[0].message.content
        
        # Заглушка для примера
        return f"[API_RESPONSE] {prompt[:100]}..."
    
    def _encode_text(self, text: str) -> Any:
        """
        Создание векторного представления через API эмбеддингов.
        
        Args:
            text: Текст для кодирования
            
        Returns:
            Векторное представление
        """
        logger.debug("Создание эмбеддинга через API")
        
        # Пример для OpenAI Embeddings API:
        # response = self.client.embeddings.create(
        #     model=self.embedding_model,
        #     input=text
        # )
        # return response.data[0].embedding
        
        # Заглушка для примера
        import numpy as np
        return np.random.rand(1536)
    
    def _compute_similarity(self, embedding1: Any, embedding2: Any) -> float:
        """
        Вычисление косинусной схожести между векторами.
        
        Args:
            embedding1: Первый вектор
            embedding2: Второй вектор
            
        Returns:
            Косинусная схожесть
        """
        import numpy as np
        
        # Косинусная схожесть
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        return float(dot_product / (norm1 * norm2))


# Пример использования
def main() -> None:
    """Пример использования суммаризаторов."""
    logging.basicConfig(level=logging.INFO)
    
    # Создание тестовых документов
    docs = [
        Document(
            content="Машинное обучение - это раздел искусственного интеллекта, "
                   "который изучает методы построения алгоритмов, способных обучаться.",
            metadata={"source": "doc1"}
        ),
        Document(
            content="Нейронные сети являются основой глубокого обучения и "
                   "используются для решения сложных задач классификации и регрессии.",
            metadata={"source": "doc2"}
        )
    ]
    
    # Пример с API (требуется реальный API ключ)
    # api_summarizer = APILLMSummarizer(
    #     model_name="gpt-4",
    #     api_key="your-api-key"
    # )
    
    # Пример с локальной моделью
    local_summarizer = LocalLLMSummarizer(
        model_name="llama-3-8b",
        model_path="/path/to/model",
        device="cuda"
    )
    
    # Абстрактивная суммаризация
    result = local_summarizer.summarize_abstractive(docs[0].content, max_length=50)
    logger.info(f"Абстрактивная суммаризация: {result.summary}")
    
    # Экстрактивная суммаризация
    result = local_summarizer.summarize_extractive(docs[0].content, num_sentences=1)
    logger.info(f"Экстрактивная суммаризация: {result.summary}")
    
    # Мульти-документная суммаризация
    result = local_summarizer.summarize_multi_document(docs, max_length=100)
    logger.info(f"Мульти-документная суммаризация: {result.summary}")
    
    # RAG запрос
    answer = local_summarizer.rag_query(
        query="Что такое машинное обучение?",
        documents=docs,
        top_k=2
    )
    logger.info(f"RAG ответ: {answer}")


if __name__ == "__main__":
    main()
