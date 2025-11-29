import os

from dotenv import load_dotenv

from src.text_summarizer.core.text2summary.api_summarizer import OpenAILLM
from src.text_summarizer.core.text2summary.local_summarizer import LocalLLM

load_dotenv()

# Пример использования
if __name__ == "__main__":
    # Локальная модель
    local_llm = LocalLLM(
        model_path="mistralai/Mistral-7B-Instruct-v0.3",
        gpu_memory_utilization=0.7,
        backend="vllm",
        device="cuda",
    )
    
    # API модель
    # api_llm = OpenAILLM(
    #     api_key=os.getenv("LLM_SECRET_TOKEN"),
    #     base_url=os.getenv("LLM_API_URL"),
    #     model="deepseek-chat",
    # )
    
    # Саммаризация диалога
    dialogue = """
    Alice: Нам нужно обсудить бюджет проекта.
    Bob: Согласен, предлагаю увеличить на 20%.
    Alice: Это слишком много, давайте 10%.
    """
    speakers_summary = local_llm.summarize_dialogue_by_speakers(
        dialogue, 
        speakers=["Alice", "Bob"]
    )
    
    # RAG генерация
    contexts = [
        "Context 1: Information about topic A",
        "Context 2: Information about topic B"
    ]
    answer = local_llm.generate_with_rag(
        query="What is topic A?",
        retrieved_contexts=contexts
    )

    print("-" * 100, answer)