@dataclass
class TrelloCardData(CardData):
    """Представление карточки Trello."""
    
    card_obj: Any  # trello.Card объект из py-trello
    
    def get_id(self) -> str:
        return self.card_obj.id


class TrelloKanbanCRUD(KanbanBoardCRUD):
    """Реализация интерфейса для работы с Trello через py-trello."""
    
    def __init__(self, api_key: str, token: str):
        """Инициализация клиента Trello.
        
        Args:
            api_key: API ключ Trello
            token: OAuth токен
        """
        from trello import TrelloClient
        
        self.client = TrelloClient(
            api_key=api_key,
            token=token,
        )
        self._boards_cache: dict[str, Any] = {}
        self._cards_cache: dict[str, Any] = {}

    def _get_board(self, board_id: str) -> Any:
        """Получить объект доски с кешированием."""
        if board_id not in self._boards_cache:
            self._boards_cache[board_id] = self.client.get_board(board_id)
        return self._boards_cache[board_id]

    def _get_card(self, card_id: str) -> Any:
        """Получить объект карточки с кешированием."""
        if card_id not in self._cards_cache:
            card = self.client.get_card(card_id)
            card.fetch()
            self._cards_cache[card_id] = card
        return self._cards_cache[card_id]

    def _invalidate_card_cache(self, card_id: str) -> None:
        """Инвалидировать кеш карточки."""
        self._cards_cache.pop(card_id, None)

    # --------- Базовые CRUD по карточкам ---------

    def list_cards(self, board_id: str, list_id: str | None = None) -> list[TrelloCardData]:
        """Получить список карточек доски или конкретной колонки."""
        board = self._get_board(board_id)
        
        if list_id:
            trello_list = self.client.get_list(list_id)
            cards = trello_list.list_cards()
        else:
            cards = board.all_cards()
        
        return [TrelloCardData(card_obj=card) for card in cards]

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
    ) -> TrelloCardData:
        """Создать новую карточку в указанной колонке."""
        trello_list = self.client.get_list(list_id)
        
        # Подготовка параметров
        card = trello_list.add_card(
            name=name,
            desc=description,
            labels=list(labels) if labels else None,
            due=due_date,
            position=position,
            assign=list(members) if members else None,
        )
        
        return TrelloCardData(card_obj=card)

    def read_card(self, card_id: str) -> TrelloCardData | None:
        """Получить карточку по ID."""
        try:
            card = self._get_card(card_id)
            return TrelloCardData(card_obj=card)
        except Exception:
            return None

    def update_card(
        self,
        card_id: str,
        name: str | None = None,
        description: str | None = None,
        closed: bool | None = None,
    ) -> TrelloCardData:
        """Обновить базовые поля карточки."""
        card = self._get_card(card_id)
        
        if name is not None:
            card.set_name(name)
        if description is not None:
            card.set_description(description)
        if closed is not None:
            card.set_closed(closed)
        
        self._invalidate_card_cache(card_id)
        return TrelloCardData(card_obj=card)

    def delete_card(self, card_id: str) -> bool:
        """Удалить карточку по ID."""
        try:
            card = self._get_card(card_id)
            card.delete()
            self._invalidate_card_cache(card_id)
            return True
        except Exception:
            return False

    # --------- Название и описание ---------

    def get_card_name(self, card: TrelloCardData) -> str:
        """Получить название карточки."""
        return card.card_obj.name

    def set_card_name(self, card_id: str, name: str) -> None:
        """Установить название карточки."""
        card = self._get_card(card_id)
        card.set_name(name)
        self._invalidate_card_cache(card_id)

    def get_card_description(self, card: TrelloCardData) -> str | None:
        """Получить описание карточки."""
        desc = card.card_obj.description
        return desc if desc else None

    def set_card_description(self, card_id: str, description: str | None) -> None:
        """Установить описание карточки."""
        card = self._get_card(card_id)
        card.set_description(description or "")
        self._invalidate_card_cache(card_id)

    # --------- Колонка (лист) карточки ---------

    def get_card_list_id(self, card: TrelloCardData) -> str:
        """Получить ID колонки, в которой находится карточка."""
        return card.card_obj.list_id

    def set_card_list(self, card_id: str, list_id: str) -> None:
        """Переместить карточку в другую колонку."""
        card = self._get_card(card_id)
        card.change_list(list_id)
        self._invalidate_card_cache(card_id)

    # --------- Ответственные ---------

    def get_card_members(self, card: TrelloCardData) -> list[str]:
        """Получить список ID всех ответственных."""
        return [member.id for member in card.card_obj.member_id]

    def get_card_main_member(self, card: TrelloCardData) -> str | None:
        """Получить ID главного ответственного (первого в списке)."""
        members = self.get_card_members(card)
        return members[0] if members else None

    def add_card_member(self, card_id: str, member_id: str) -> None:
        """Добавить участника к карточке."""
        card = self._get_card(card_id)
        card.assign(member_id)
        self._invalidate_card_cache(card_id)

    def remove_card_member(self, card_id: str, member_id: str) -> None:
        """Удалить участника из карточки."""
        card = self._get_card(card_id)
        card.unassign(member_id)
        self._invalidate_card_cache(card_id)

    def set_card_members(self, card_id: str, member_ids: Iterable[str]) -> None:
        """Заменить список ответственных."""
        card = self._get_card(card_id)
        
        # Удаляем текущих участников
        current_members = [member.id for member in card.member_id]
        for member_id in current_members:
            card.unassign(member_id)
        
        # Добавляем новых
        for member_id in member_ids:
            card.assign(member_id)
        
        self._invalidate_card_cache(card_id)

    # --------- Метки (Labels) ---------

    def get_card_labels(self, card: TrelloCardData) -> list[str]:
        """Получить список ID меток карточки."""
        return [label.id for label in card.card_obj.labels]

    def add_card_label(self, card_id: str, label_id: str) -> None:
        """Добавить метку к карточке."""
        card = self._get_card(card_id)
        board = self._get_board(card.board_id)
        label = self.client.get_label(label_id, board.id)
        card.add_label(label)
        self._invalidate_card_cache(card_id)

    def remove_card_label(self, card_id: str, label_id: str) -> None:
        """Удалить метку с карточки."""
        card = self._get_card(card_id)
        board = self._get_board(card.board_id)
        label = self.client.get_label(label_id, board.id)
        card.remove_label(label)
        self._invalidate_card_cache(card_id)

    # --------- Чек‑листы ---------

    def get_card_checklists(self, card: TrelloCardData) -> list[dict[str, Any]]:
        """Получить чек‑листы карточки."""
        card.card_obj.fetch_checklists()
        checklists = []
        
        for checklist in card.card_obj.checklists:
            items = [
                {
                    'id': item['id'],
                    'name': item['name'],
                    'checked': item['checked'],
                }
                for item in checklist.items
            ]
            checklists.append({
                'id': checklist.id,
                'name': checklist.name,
                'items': items,
            })
        
        return checklists

    def add_checklist(
        self,
        card_id: str,
        name: str,
        items: Iterable[tuple[str, bool]] | None = None,
    ) -> str:
        """Создать чек‑лист с опциональными пунктами."""
        card = self._get_card(card_id)
        
        item_names = []
        item_states = []
        
        if items:
            for item_name, checked in items:
                item_names.append(item_name)
                item_states.append(checked)
        
        checklist = card.add_checklist(
            title=name,
            items=item_names,
            itemstates=item_states if item_states else None,
        )
        
        self._invalidate_card_cache(card_id)
        return checklist.id

    def add_checklist_item(
        self,
        checklist_id: str,
        name: str,
        checked: bool = False,
    ) -> str:
        """Добавить пункт в чек-лист."""
        # Получаем чек-лист через API
        json_obj = self.client.fetch_json(
            f'/checklists/{checklist_id}',
            http_method='GET',
        )
        
        from trello import Checklist
        checklist = Checklist(self.client, checked=False, obj=json_obj)
        
        item = checklist.add_checklist_item(name, checked)
        return item['id']

    def update_checklist_item(
        self,
        checklist_id: str,
        item_id: str,
        name: str | None = None,
        checked: bool | None = None,
    ) -> None:
        """Обновить пункт чек-листа."""
        params = {}
        
        if name is not None:
            params['name'] = name
        if checked is not None:
            params['state'] = 'complete' if checked else 'incomplete'
        
        if params:
            self.client.fetch_json(
                f'/checklists/{checklist_id}/checkItems/{item_id}',
                http_method='PUT',
                post_args=params,
            )

    def delete_checklist(self, checklist_id: str) -> None:
        """Удалить чек‑лист."""
        json_obj = self.client.fetch_json(
            f'/checklists/{checklist_id}',
            http_method='GET',
        )
        
        from trello import Checklist
        checklist = Checklist(self.client, checked=False, obj=json_obj)
        checklist.delete()

    # --------- Комментарии ---------

    def get_card_comments(self, card: TrelloCardData) -> list[dict[str, Any]]:
        """Получить список комментариев карточки."""
        card.card_obj.fetch_comments()
        
        comments = []
        for comment in card.card_obj.comments:
            comments.append({
                'id': comment['id'],
                'text': comment['data']['text'],
                'date': comment['date'],
                'member_id': comment.get('idMemberCreator', ''),
            })
        
        return comments

    def add_comment(self, card_id: str, text: str) -> str:
        """Добавить комментарий к карточке."""
        card = self._get_card(card_id)
        result = card.comment(text)
        self._invalidate_card_cache(card_id)
        return result['id']

    def update_comment(self, card_id: str, comment_id: str, text: str) -> None:
        """Обновить текст комментария."""
        card = self._get_card(card_id)
        card.update_comment(comment_id, text)
        self._invalidate_card_cache(card_id)

    def delete_comment(self, card_id: str, comment_id: str) -> None:
        """Удалить комментарий."""
        card = self._get_card(card_id)
        
        # Fetch comments to get the comment object
        card.fetch_comments()
        comment_obj = next((c for c in card.comments if c['id'] == comment_id), None)
        
        if comment_obj:
            card.delete_comment(comment_obj)
            self._invalidate_card_cache(card_id)

    # --------- Списки (колонки) ---------

    def list_lists(self, board_id: str) -> list[dict[str, Any]]:
        """Получить все списки (колонки) доски."""
        board = self._get_board(board_id)
        lists = board.all_lists()
        
        return [
            {
                'id': lst.id,
                'name': lst.name,
                'closed': lst.closed,
                'pos': lst.pos,
            }
            for lst in lists
        ]

    def create_list(self, board_id: str, name: str, position: str | None = None) -> str:
        """Создать новую колонку на доске."""
        board = self._get_board(board_id)
        new_list = board.add_list(name, pos=position)
        return new_list.id
