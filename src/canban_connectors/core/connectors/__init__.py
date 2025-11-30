"""Модуль, содержащий имплементации коннекторов к канбан-доскам."""

from .kaiten_connector import KaitenKanbanBoard
from .trello_connector import TrelloKanbanBoard

__all__ = [
    "KaitenKanbanBoard",
    "TrelloKanbanBoard",
]
