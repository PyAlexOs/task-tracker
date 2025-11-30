"""Интерфейс базового коннектора к канбан-доске."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class KanbanBoardBase(ABC):
    """Абстрактный интерфейс для работы с канбан-досками."""

    def __init__(self, **kwargs: Any):
        """Конструктор класса."""

        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get_all_boards(self) -> list[dict[str, Any]]:
        """Получить список всех досок, доступных пользователю.

        Returns:
            list[dict[str, Any]]: Список словарей с информацией о досках. Каждый словарь содержит
                базовые поля доски (id, name, url и т.д.).

        Raises:
            Exception: При ошибке получения данных от API.
        """

    @abstractmethod
    def get_board_id_by_name(self, board_name: str) -> str | None:
        """Получить ID доски по её названию.

        Args:
            board_name (str): Название доски для поиска.

        Returns:
            str | None: ID доски, если доска найдена, иначе None.

        Raises:
            Exception: При ошибке получения данных от API.
        """

    @abstractmethod
    def get_all_cards(self, board_id: str) -> list[dict[str, Any]]:
        """Получить список всех карточек на доске.

        Args:
            board_id (str): Уникальный идентификатор доски.

        Returns:
            list[dict[str, Any]]: Список словарей с информацией о карточках. Каждый словарь содержит
                базовые поля карточки (id, name, list_id и т.д.).

        Raises:
            Exception: При ошибке получения данных от API.
        """

    @abstractmethod
    def get_board_lists(self, board_id: str) -> list[dict[str, Any]]:
        """Получить список всех столбцов (листов) на доске.

        Args:
            board_id (str): Уникальный идентификатор доски.

        Returns:
            list[dict[str, Any]]: Список словарей с информацией о столбцах. Каждый словарь содержит
                поля столбца (id, name, position и т.д.).

        Raises:
            Exception: При ошибке получения данных от API.
        """

    @abstractmethod
    def get_board_labels(self, board_id: str) -> list[dict[str, Any]]:
        """Получить список всех меток, доступных на доске.

        Args:
            board_id (str): Уникальный идентификатор доски.

        Returns:
            list[dict[str, Any]]: Список словарей с информацией о метках. Каждый словарь содержит
                поля метки (id, name, color и т.д.).

        Raises:
            Exception: При ошибке получения данных от API.
        """

    @abstractmethod
    def get_board_members(self, board_id: str) -> list[dict[str, Any]]:
        """Получить список всех участников доски.

        Args:
            board_id (str): Уникальный идентификатор доски.

        Returns:
            list[dict[str, Any]]: Список словарей с информацией об участниках.
                Каждый словарь содержит поля пользователя (id, username, full_name и т.д.).

        Raises:
            Exception: При ошибке получения данных от API.
        """

    @abstractmethod
    def get_card_details(self, card_id: str) -> dict[str, Any]:
        """Получить подробную информацию о карточке.

        Args:
            card_id (str): Уникальный идентификатор карточки.

        Returns:
            dict[str, Any]: Словарь с полной информацией о карточке, включающий название,
                описание, столбец, ответственных, метки, чек-листы, даты и комментарии.

        Raises:
            Exception: При ошибке получения данных от API.
        """

    @abstractmethod
    def create_card(
        self,
        list_id: str,
        name: str,
        description: str | None = None,
        labels: list[str] | None = None,
        members: list[str] | None = None,
        due_date: datetime | None = None,
        checklist_items: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Создать новую карточку с указанными полями.

        Args:
            list_id (str): ID столбца, в котором создается карточка.
            name (str): Название карточки.
            description (str | None): Описание карточки. Defaults to None.
            labels (list[str] | None): Список ID меток для карточки. Defaults to None.
            members (list[str] | None): Список ID ответственных пользователей. Defaults to None.
            due_date (datetime | None): Срок выполнения карточки. Defaults to None.
            checklist_items (dict[str, list[str]] | None): Словарь чек-листов в формате
                {название_чеклиста: [пункт1, пункт2, ...]}. Defaults to None.

        Returns:
            dict[str, Any]: Словарь с информацией о созданной карточке (id, name, url и т.д.).

        Raises:
            Exception: При ошибке создания карточки.
        """

    @abstractmethod
    def move_card_to_list(self, card_id: str, list_id: str) -> bool:
        """Переместить карточку в другой столбец.

        Args:
            card_id (str): ID карточки для перемещения.
            list_id (str): ID целевого столбца.

        Returns:
            bool: True, если перемещение прошло успешно, иначе False.

        Raises:
            Exception: При ошибке перемещения карточки.
        """

    @abstractmethod
    def add_member_to_card(self, card_id: str, member_id: str) -> bool:
        """Добавить ответственного к карточке.

        Args:
            card_id (str): ID карточки.
            member_id (str): ID пользователя, которого нужно добавить как ответственного.

        Returns:
            bool: True, если добавление прошло успешно, иначе False.

        Raises:
            Exception: При ошибке добавления ответственного.
        """

    @abstractmethod
    def add_comment(self, card_id: str, comment_text: str) -> bool:
        """Написать комментарий к карточке.

        Args:
            card_id (str): ID карточки.
            comment_text (str): Текст комментария.

        Returns:
            bool: True, если комментарий добавлен успешно, иначе False.

        Raises:
            Exception: При ошибке добавления комментария.
        """

    @abstractmethod
    def modify_checklist(
        self,
        card_id: str,
        checklist_name: str,
        item_name: str | None = None,
        add_item: bool = False,  # TODO может быть здесь это не нужно, можно просто сначала смотреть наличие
        checked: bool | None = None,
    ) -> bool:
        """Изменить чек-лист карточки.

        Метод позволяет добавить новый пункт в чек-лист или изменить статус
        существующего пункта (отметить как выполненный/невыполненный).

        Args:
            card_id (str): ID карточки.
            checklist_name (str): Название чек-листа.
            item_name (str | None): Название пункта чек-листа. Defaults to None.
            add_item (bool): Флаг добавления нового пункта. Если True, то создается
                новый пункт с названием item_name. Если False, то изменяется
                статус существующего пункта. Defaults to False.
            checked (bool | None): Статус выполнения пункта - True (выполнен) или
                False (не выполнен). Используется только при изменении
                существующего пункта (опционально).

        Returns:
            bool: True, если операция прошла успешно, иначе False.

        Raises:
            Exception: При ошибке изменения чек-листа.
        """
