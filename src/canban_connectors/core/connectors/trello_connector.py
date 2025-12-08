"""Имплементация коннектора к канбан-доске Trello."""

from datetime import datetime
from typing import Any

from trello import TrelloClient

from ..base import KanbanBoardBase


class TrelloKanbanBoard(KanbanBoardBase):
    """Реализация интерфейса для работы с Trello.

    Класс предоставляет методы для взаимодействия с досками Trello
    через библиотеку py-trello.
    """

    def __init__(self, api_key: str, token: str):
        """Инициализация клиента Trello.

        Args:
            api_key (str): API ключ Trello (https://trello.com/app-key).
            token (str): OAuth токен для авторизации.

        Raises:
            Exception: При ошибке инициализации клиента.
        """

        super().__init__()
        self._client = TrelloClient(api_key=api_key, token=token)

    def get_all_boards(self) -> list[dict[str, Any]]:
        boards = self._client.list_boards()

        return [
            {
                "id": board.id,
                "name": board.name,
                "url": board.url,
                "closed": board.closed,
                "description": board.description if hasattr(board, "description") else None,
            }
            for board in boards
        ]

    def get_board_id_by_name(self, board_name: str) -> str | None:
        try:
            boards = self._client.list_boards()
            
            for board in boards:
                if board.name == board_name:
                    return str(board.id)
            
            self.logger.warning(f"Доска с названием '{board_name}' не найдена.")
        
        except Exception:
            self.logger.exception("Ошибка при поиске доски по названию.")

        return None


    def get_all_cards(self, board_id: str) -> list[dict[str, Any]]:
        board = self._client.get_board(board_id)
        cards = board.all_cards()

        return [
            {
                "id": card.id,
                "name": card.name,
                "list_id": card.list_id,
                "list_name": card.trello_list.name if hasattr(card, "trello_list") else None,
                "url": card.url,
                "short_id": card.idShort,
            }
            for card in cards
        ]

    def get_board_lists(self, board_id: str) -> list[dict[str, Any]]:
        board = self._client.get_board(board_id)
        lists = board.list_lists()

        return [
            {
                "id": trello_list.id,
                "name": trello_list.name,
                "closed": trello_list.closed,
                "pos": trello_list.pos,
                "subscribed": trello_list.subscribed,
            }
            for trello_list in lists
        ]

    def get_board_labels(self, board_id: str) -> list[dict[str, Any]]:
        board = self._client.get_board(board_id)
        labels = board.get_labels()
        return [{"id": label.id, "name": label.name, "color": label.color} for label in labels]

    def get_board_members(self, board_id: str) -> list[dict[str, Any]]:
        board = self._client.get_board(board_id)
        members = board.get_members()

        return [
            {
                "id": member.id,
                "username": member.username,
                "full_name": member.full_name,
                "initials": member.initials,
            }
            for member in members
        ]

    def get_card_details(self, card_id: str) -> dict[str, Any]:
        card = self._client.get_card(card_id)
        card.fetch(eager=True)

        # Получаем информацию о метках
        labels = [
            {"id": label.id, "name": label.name, "color": label.color} for label in card.labels
        ]

        # Получаем информацию об ответственных
        members = []
        if card.member_id:
            for member_id in card.member_id:
                try:
                    member = self._client.get_member(member_id)
                    members.append({
                        "id": member.id,
                        "full_name": member.full_name,
                        "username": member.username
                    })
                except Exception as e:
                    self.logger.warning(f"Не удалось загрузить участника {member_id}: {e}")

        # Получаем чек-листы
        checklists_data = []
        for checklist in card.checklists:
            items = [
                {"name": item["name"], "checked": item["checked"], "id": item["id"]}
                for item in checklist.items
            ]
            checklists_data.append({"id": checklist.id, "name": checklist.name, "items": items})

        # Получаем комментарии
        comments = [
            {
                "id": comment["id"],
                "text": comment["data"]["text"],
                "date": comment["date"],
                "member_creator": comment["memberCreator"]["fullName"],
            }
            for comment in card.comments
        ]

        return {
            "id": card.id,
            "name": card.name,
            "description": card.description,
            "list_id": card.list_id,
            "list_name": card.trello_list.name if hasattr(card, "trello_list") else None,
            "labels": labels,
            "members": members,
            "due_date": card.due,
            "checklists": checklists_data,
            "comments": comments,
            "url": card.url,
            "created_date": card.created_date,
        }

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
        try:
            trello_list = self._client.get_list(list_id)
            
            # Создаем карточку БЕЗ меток (они будут добавлены потом)
            card = trello_list.add_card(
                name=name,
                desc=description or "",
            )
            
            # Добавляем due date если указан
            if due_date:
                card.set_due(due_date)
            
            # Добавляем метки через add_label
            if labels:
                for label_id in labels:
                    try:
                        card.add_label(card.board.get_label(label_id))
                    except Exception as e:
                        self.logger.warning(f"Не удалось добавить метку {label_id}: {e}")
            
            # Добавляем ответственных
            if members:
                for member_id in members:
                    try:
                        member = self._client.get_member(member_id)
                        card.add_member(member)
                    except Exception as e:
                        self.logger.warning(f"Не удалось добавить участника {member_id}: {e}")
            
            # Добавляем чек-листы
            if checklist_items:
                for checklist_name, items in checklist_items.items():
                    try:
                        card.add_checklist(checklist_name, items)
                    except Exception as e:
                        self.logger.warning(f"Не удалось добавить чек-лист {checklist_name}: {e}")

        except Exception:
            self.logger.exception("Ошибка при создании карточки.")
            raise

        else:
            return {"id": card.id, "name": card.name, "url": card.url}

    def move_card_to_list(self, card_id: str, list_id: str) -> bool:
        try:
            card = self._client.get_card(card_id)
            card.change_list(list_id)

        except Exception:
            self.logger.exception("Ошибка при перемещении карточки.")
            return False

        else:
            return True

    def add_member_to_card(self, card_id: str, member_id: str) -> bool:
        try:
            card = self._client.get_card(card_id)
            member = self._client.get_member(member_id)
            card.add_member(member)

        except Exception:
            self.logger.exception("Ошибка при добавлении ответственного.")
            return False

        else:
            return True

    def add_comment(self, card_id: str, comment_text: str) -> bool:
        try:
            card = self._client.get_card(card_id)
            card.comment(comment_text)

        except Exception:
            self.logger.exception("Ошибка при добавлении комментария.")
            return False

        else:
            return True

    def modify_checklist(
        self,
        card_id: str,
        checklist_name: str,
        item_name: str | None = None,
         add_item: bool = False,
        checked: bool | None = None,
    ) -> bool:
        try:
            card = self._client.get_card(card_id)
            card.fetch(eager=True)

            # Находим нужный чек-лист
            checklist = None
            for cl in card.checklists:
                if cl.name == checklist_name:
                    checklist = cl
                    break

            if not checklist:
                # Если чек-лист не найден, создаем новый
                if add_item and item_name:
                    card.add_checklist(checklist_name, [item_name])
                    return True
                return False

            if add_item and item_name:
                # Добавляем новый пункт
                checklist.add_checklist_item(item_name)
            elif item_name and checked is not None:
                # Изменяем статус существующего пункта
                checklist.set_checklist_item(item_name, checked)

        except Exception as e:
            print(f"Ошибка при изменении чек-листа: {e}")
            return False
        
        else:
            return True
