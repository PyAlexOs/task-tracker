from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, TypeVar

# ========== Абстрактные базовые классы ==========


class CardData(ABC):
    """Абстрактный тип, обозначающий представление карточки канбан-доски.

    Конкретные реализации коннекторов могут использовать свои датаклассы / Pydantic‑модели,
    которые наследуются от этого абстрактного типа.
    """

    @abstractmethod
    def get_id(self) -> str:
        """Получить ID карточки."""


TCardData = TypeVar("TCardData", bound=CardData)


class KanbanBoardCRUD(ABC):
    """Абстрактный интерфейс операций с канбан‑досками и карточками."""

    # --------- Базовые CRUD по карточкам ---------

    @abstractmethod
    def list_cards(self, board_id: str, list_id: str | None = None) -> list[CardData]:
        """Получить список карточек доски или конкретной колонки.

        Args:
            board_id: ID доски
            list_id: ID колонки (если None, возвращаются карточки всей доски)
        """

    @abstractmethod
    def create_card(
        self,
        board_id: str,
        list_id: str,
        name: str,
        description: str | None = None,
        members: Iterable[str] | None = None,
        labels: Iterable[str] | None = None,
        due_date: str | None = None,
        position: str | int | None = None,
    ) -> CardData:
        """Создать новую карточку в указанной колонке.

        Args:
            board_id: ID доски
            list_id: ID колонки
            name: Название карточки
            description: Описание карточки
            members: ID участников для назначения
            labels: ID меток
            due_date: Срок выполнения
            position: Позиция ('top', 'bottom' или число)
        """

    @abstractmethod
    def read_card(self, card_id: str) -> CardData | None:
        """Получить карточку по ID.

        Args:
            card_id: ID карточки
        """

    @abstractmethod
    def update_card(
        self,
        card_id: str,
        name: str | None = None,
        description: str | None = None,
        closed: bool | None = None,
    ) -> CardData:
        """Обновить базовые поля карточки.

        Args:
            card_id: ID карточки
            name: Новое название
            description: Новое описание
            closed: Статус закрытия (архивирования)
        """

    @abstractmethod
    def delete_card(self, card_id: str) -> bool:
        """Удалить карточку по ID.

        Args:
            card_id: ID карточки

        Returns:
            True, если карточка была удалена
        """

    # --------- Название и описание ---------

    @abstractmethod
    def get_card_name(self, card: CardData) -> str:
        """Получить название карточки."""

    @abstractmethod
    def set_card_name(self, card_id: str, name: str) -> None:
        """Установить название карточки."""

    @abstractmethod
    def get_card_description(self, card: CardData) -> str | None:
        """Получить описание карточки."""

    @abstractmethod
    def set_card_description(self, card_id: str, description: str | None) -> None:
        """Установить описание карточки."""

    # --------- Колонка (лист) карточки ---------

    @abstractmethod
    def get_card_list_id(self, card: CardData) -> str:
        """Получить ID колонки (листа), в которой сейчас находится карточка."""

    @abstractmethod
    def set_card_list(self, card_id: str, list_id: str) -> None:
        """Переместить карточку в другую колонку."""

    # --------- Ответственные ---------

    @abstractmethod
    def get_card_members(self, card: CardData) -> list[str]:
        """Получить список ID всех ответственных (участников) карточки."""

    @abstractmethod
    def get_card_main_member(self, card: CardData) -> str | None:
        """Получить ID главного ответственного.

        Возвращает первого участника из списка или None.
        """

    @abstractmethod
    def add_card_member(self, card_id: str, member_id: str) -> None:
        """Добавить участника к карточке."""

    @abstractmethod
    def remove_card_member(self, card_id: str, member_id: str) -> None:
        """Удалить участника из карточки."""

    @abstractmethod
    def set_card_members(self, card_id: str, member_ids: Iterable[str]) -> None:
        """Заменить список ответственных полностью."""

    # --------- Метки (Labels) ---------

    @abstractmethod
    def get_card_labels(self, card: CardData) -> list[str]:
        """Получить список ID меток карточки."""

    @abstractmethod
    def add_card_label(self, card_id: str, label_id: str) -> None:
        """Добавить метку к карточке."""

    @abstractmethod
    def remove_card_label(self, card_id: str, label_id: str) -> None:
        """Удалить метку с карточки."""

    # --------- Чек‑листы ---------

    @abstractmethod
    def get_card_checklists(self, card: CardData) -> list[dict[str, Any]]:
        """Получить чек‑листы карточки.

        Возвращает список словарей с информацией о чек-листах.
        Формат: [{'id': str, 'name': str, 'items': [{'id': str, 'name': str, 'checked': bool}]}]
        """

    @abstractmethod
    def add_checklist(
        self,
        card_id: str,
        name: str,
        items: Iterable[tuple[str, bool]] | None = None,
    ) -> str:
        """Создать чек‑лист с опциональными пунктами.

        Args:
            card_id: ID карточки
            name: Название чек-листа
            items: Список кортежей (название_пункта, выполнен)

        Returns:
            ID созданного чек-листа
        """

    @abstractmethod
    def add_checklist_item(
        self,
        checklist_id: str,
        name: str,
        checked: bool = False,
    ) -> str:
        """Добавить пункт в чек-лист.

        Returns:
            ID созданного пункта
        """

    @abstractmethod
    def update_checklist_item(
        self,
        checklist_id: str,
        item_id: str,
        name: str | None = None,
        checked: bool | None = None,
    ) -> None:
        """Обновить пункт чек-листа."""

    @abstractmethod
    def delete_checklist(self, checklist_id: str) -> None:
        """Удалить чек‑лист."""

    # --------- Комментарии ---------

    @abstractmethod
    def get_card_comments(self, card: CardData) -> list[dict[str, Any]]:
        """Получить список комментариев карточки.

        Формат: [{'id': str, 'text': str, 'date': str, 'member_id': str}]
        """

    @abstractmethod
    def add_comment(self, card_id: str, text: str) -> str:
        """Добавить комментарий к карточке.

        Returns:
            ID созданного комментария
        """

    @abstractmethod
    def update_comment(self, card_id: str, comment_id: str, text: str) -> None:
        """Обновить текст комментария."""

    @abstractmethod
    def delete_comment(self, card_id: str, comment_id: str) -> None:
        """Удалить комментарий."""

    # --------- Списки (колонки) ---------

    @abstractmethod
    def list_lists(self, board_id: str) -> list[dict[str, Any]]:
        """Получить все списки (колонки) доски.

        Формат: [{'id': str, 'name': str, 'closed': bool, 'pos': float}]
        """

    @abstractmethod
    def create_list(self, board_id: str, name: str, position: str | None = None) -> str:
        """Создать новую колонку на доске.

        Args:
            board_id: ID доски
            name: Название колонки
            position: Позиция ('top', 'bottom')

        Returns:
            ID созданной колонки
        """

    # --------- Удобные хелперы ---------

    def move_card(self, card_id: str, dst_list_id: str) -> None:
        """Более говорящий синоним set_card_list."""
        self.set_card_list(card_id, dst_list_id)

    def rename_card(self, card_id: str, new_name: str) -> None:
        """Переименовать карточку."""
        self.set_card_name(card_id, new_name)

    def archive_card(self, card_id: str) -> None:
        """Архивировать карточку (закрыть)."""
        self.update_card(card_id, closed=True)

    def unarchive_card(self, card_id: str) -> None:
        """Разархивировать карточку (открыть)."""
        self.update_card(card_id, closed=False)
