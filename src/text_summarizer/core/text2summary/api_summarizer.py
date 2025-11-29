import os

from openai import OpenAI

from src.text_summarizer.core.base import BaseLLM


class OpenAILLM(BaseLLM):
    """
    Реализация для работы через OpenAI API (или совместимые API).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 2048,
    ):
        """

        Args:
            api_key: API ключ (если None, берется из OPENAI_API_KEY)
            base_url: Базовый URL API (для совместимых API)
            model: Название модели
            max_tokens: Максимальное количество токенов
        """

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )
        self.model = model
        self.max_tokens = max_tokens

    def _generate(
        self, 
        prompt: str, 
        max_tokens: int | None = None,
        temperature: float = 0.7
    ) -> str:
        """Внутренний метод для генерации через API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def summarize_text(self, text: str, max_length: int | None = None) -> str:
        prompt = f"""Summarize the following text concisely:

Text: {text}

Summary:"""
        return self._generate(prompt, max_length)

    def summarize_dialogue_by_speakers(
        self, 
        dialogue: str, 
        speakers: list[str],
        max_length: int | None = None
    ) -> dict[str, str]:
        summaries = {}
        for speaker in speakers:
            prompt = f"""Summarize what {speaker} said in the following dialogue:
                Dialogue:
                {dialogue}

                Summary of {speaker}'s contributions:"""
            summaries[speaker] = self._generate(prompt, max_length)
        return summaries

    def summarize_by_topics(
        self,
        text: str,
        topics: list[str],
        max_length: int | None = None
    ) -> dict[str, str]:
        summaries = {}
        for topic in topics:
            prompt = f"""Extract and summarize information about "{topic}" from the following text:

                Text: {text}

                Summary about {topic}:"""
            summaries[topic] = self._generate(prompt, max_length)
        return summaries

    def find_untouched_topics(
        self, 
        text: str, 
        expected_topics: list[str]
    ) -> list[str]:
        topics_str = ", ".join(expected_topics)
        prompt = f"""Given the following text and list of topics, identify which topics were NOT discussed or mentioned.

            Text: {text}

            Expected topics: {topics_str}

            List ONLY the topics that were NOT touched upon (comma-separated):"""
        
        response = self._generate(prompt, max_tokens=200)
        untouched = [topic.strip() for topic in response.split(",")]
        return [t for t in untouched if t in expected_topics]

    def generate_with_rag(
        self, 
        query: str, 
        retrieved_contexts: list[str],
        max_length: int | None = None,
        temperature: float = 0.7
    ) -> str:
        contexts_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(retrieved_contexts)])
        
        prompt = f"""Answer the question based on the following context information. Use only the information provided.

            Context:
            {contexts_text}

            Question: {query}

            Answer:"""
        
        return self._generate(prompt, max_length, temperature)
