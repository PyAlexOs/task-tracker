from vllm import LLM, SamplingParams

from src.text_summarizer.core.base import BaseLLM


class LocalLLM(BaseLLM):
    """
    Реализация для локальной LLM с использованием llama.cpp или vLLM.
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "vllm",
        device: str = "cuda",
        max_tokens: int = 2048,
        **kwargs
    ):
        """
        Args:
            model_path: Путь к файлу модели (GGUF для llama.cpp)
            backend: Бэкенд для инференса ("llama.cpp" или "vllm")
            device: Устройство ("cuda" или "cpu")
            max_tokens: Максимальное количество токенов
            **kwargs: Дополнительные параметры для бэкенда
        """
        self.model_path = model_path
        self.backend = backend
        self.device = device
        self.max_tokens = max_tokens

        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.9),
        )

    def _generate(self, prompt: str, max_tokens: int | None = None, temperature: float = 0.7) -> str:
        """Внутренний метод для генерации текста."""
        max_tokens = max_tokens or self.max_tokens
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

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
