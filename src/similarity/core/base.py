"""Функциональность для извлечения семантических эмбеддингов текстов
и получения их схожести, а также для ранкинга корпуса текстов в отношении эталонного.
"""

import logging
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer


class SimilarityPipeline:
    """Класс для вычисления схожести текстов с использованием BERT-эмбеддингов
    и косинусного расстояния.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        reduce_mem_consumption: bool = True,
    ):
        """Конструктор класса.

        Args:
            model_name (str): Название предобученной модели из HuggingFace.
            device (str | None): Устройство для вычислений ("cuda", "mps", "cpu", "cuda:0" и т.д.).
                Defaults to "cuda".
            reduce_mem_consumption (bool): Если True, модель загружается и работает в float16
                для экономии памяти и ускорения вычислений. Defaults to True.
        """

        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = self._get_device(device)

        self.reduce_mem_consumption = reduce_mem_consumption
        self.dtype = torch.float32
        if reduce_mem_consumption:
            self.dtype = torch.float16

        self.logger.debug(
            "Используется устройство %s, точность вычислений %s.",
            self.device,
            self.dtype,
        )

        self.logger.debug("Загрузка токенизатора: %s.", model_name)
        self.tokenizer: PreTrainedTokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(model_name),
        )

        self.logger.debug("Загрузка модели: %s.", model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            dtype=self.dtype,
        ).to(self.device)

        self.model.eval()
        self.logger.debug("Модель успешно загружена.")

    def _get_device(self, device: str) -> torch.device:
        """
        Определение устройства для вычислений.

        Args:
            device (str): Указанное устройство или None для автовыбора

        Returns:
            torch.device объект
        """

        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")
    
    # ===================================== #
    # Функционал для извлечения эмбеддингов #
    # ===================================== #

    def _mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling для получения эмбеддинга предложения из токенов.

        Args:
            model_output (torch.Tensor): Выход модели BERT.
            attention_mask (torch.Tensor): Маска внимания.

        Returns:
            torch.Tensor: Усредненный эмбеддинг предложения.
        """

        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1),
            min=1e-9,
        )

    def get_embedding(self, text: str) -> NDArray[Any]:
        """Получение эмбеддинга для одного текста.

        Args:
            text (str): Входной текст.

        Returns:
            NDArray[Any]: Нормализованный вектор-эмбеддинг.
        """

        self.logger.debug(
            "Получение эмбеддинга для текста длиной %d символов.",
            len(text),
        )

        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # Используем autocast для mixed precision инференса
            with torch.autocast(
                device_type=self.device.type, dtype=self.dtype, enabled=self.reduce_mem_consumption
            ):
                model_output = self.model(**encoded_input)

        sentence_embedding = self._mean_pooling(
            model_output,
            encoded_input["attention_mask"],
        )

        # Нормализация для корректного косинусного расстояния
        sentence_embedding = torch.nn.functional.normalize(
            sentence_embedding,
            p=2,
            dim=1,
        )

        return cast(NDArray[Any], sentence_embedding.cpu().float().numpy()[0])

    def get_embeddings_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> NDArray[Any]:
        """Получение эмбеддингов для батча текстов (более эффективно).

        Args:
            texts (list[str]): Список текстов
            batch_size (int | None): Размер батча для обработки.
                Если None, обрабатываются все тексты сразу.

        Returns:
            NDArray[Any]: Массив нормализованных эмбеддингов.
        """

        self.logger.debug("Получение эмбеддингов для %d текстов.", len(texts))
        if batch_size is None:
            return self._process_batch(texts)

        # Обработка по батчам для больших объемов данных
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            self.logger.debug(
                "Обработка батча %d:%d текстов",
                i // batch_size + 1,
                len(batch),
            )

            batch_embeddings = self._process_batch(batch)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def _process_batch(self, texts: list[str]) -> NDArray[Any]:
        """Внутренний метод для обработки одного батча текстов.

        Args:
            texts (list[str]): Список текстов.

        Returns:
            NDArray[Any]: Массив нормализованных эмбеддингов.
        """

        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type, dtype=self.dtype, enabled=self.reduce_mem_consumption
            ):
                model_output = self.model(**encoded_input)

        sentence_embeddings = self._mean_pooling(
            model_output,
            encoded_input["attention_mask"],
        )

        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings,
            p=2,
            dim=1,
        )

        return sentence_embeddings.cpu().float().numpy()

    # ==================================================== #
    # Функционал вычисления семантической близости текстов #
    # ==================================================== #
    #
    # 1. cosine_similarity - семантическая схожесть двух текстов.
    #
    # 2. get_similarity_matrix - матрица семантической схожести для переданных текстов.
    #
    # 3. rank_texts - ранжированные первые k самых похожих на эталонный текст.
    #
    # 4. find_most_similar - ранжированные, похожие более чем на threshold на эталонный текст.

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Вычисление косинусного сходства между двумя текстами.

        Args:
            text1 (str): Первый текст
            text2 (str): Второй текст

        Returns:
            float: Скор схожести от 0 до 1.
        """

        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        # scipy.cosine возвращает расстояние, преобразуем в similarity
        return float(1 - cosine(emb1, emb2))
    
    def get_similarity_matrix(self, texts: list[str]) -> NDArray[Any]:
        """Вычисление матрицы попарных схожестей для списка текстов.

        Args:
            texts (list[str]): Список текстов.

        Returns:
            NDArray[Any]: Матрица схожестей размера (n, n).
        """

        self.logger.debug("Вычисление матрицы схожестей для %d текстов", len(texts))
        embeddings = self.get_embeddings_batch(texts)

        # Матричное умножение нормализованных векторов = косинусное сходство
        return cast(
            NDArray[Any],
            np.dot(embeddings, embeddings.T),
        )

    def rank_texts(
        self,
        reference_text: str,
        candidate_texts: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, str, float]]:
        """Ранжирование списка текстов относительно эталонного текста.

        Args:
            reference_text (str): Эталонный текст для сравнения.
            candidate_texts (list[str]): Список текстов-кандидатов.
            top_k (int | None): Количество топовых результатов (None - все).

        Returns:
            list[tuple[int, str, float]]: Список кортежей (индекс, текст, скор),
                отсортированный по убыванию скора.
        """

        self.logger.debug(
            "Ранжирование текстов относительно эталонного",
            len(candidate_texts),
        )

        reference_emb = self.get_embedding(reference_text)
        candidate_embs = self.get_embeddings_batch(candidate_texts)

        similarities: list[tuple[int, str, float]] = []
        for idx, candidate_emb in enumerate(candidate_embs):
            similarity = 1 - cosine(reference_emb, candidate_emb)
            similarities.append((idx, candidate_texts[idx], float(similarity)))

        # Сортировка по убыванию
        similarities.sort(key=lambda x: x[-1], reverse=True)

        if top_k is not None:
            return similarities[:top_k]
        return similarities

    def find_most_similar(
        self, query_text: str, corpus_texts: list[str], threshold: float | None = None,
    ) -> list[tuple[int, str, float]]:
        """Поиск наиболее похожих текстов из корпуса по отношению к запросу.

        Args:
            query_text (str): Текст запроса.
            corpus_texts (list[str]): Корпус текстов для поиска.
            threshold (float | None): Минимальный порог схожести для фильтрации (0-1).

        Returns:
            list[tuple[int, str, float]]: Список кортежей (индекс, текст, скор),
                отсортированный по убыванию.
        """

        self.logger.debug(f"Поиск похожих текстов в корпусе из {len(corpus_texts)} документов")
        results = self.rank_texts(query_text, corpus_texts)

        if threshold is not None:
            results = [
                (idx, text, score)
                for idx, text, score in results
                if score >= threshold
            ]

        return results
