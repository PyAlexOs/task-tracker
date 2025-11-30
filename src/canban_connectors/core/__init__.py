"""Модуль, содержащий функциональность для работы с канбан-досками."""

from .base import KanbanBoardBase
from .connectors import KaitenKanbanBoard, TrelloKanbanBoard

__all__ = [
    "KanbanBoardBase",
    "KaitenKanbanBoard",
    "TrelloKanbanBoard",
]
