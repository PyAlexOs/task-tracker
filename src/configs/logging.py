"""Конфигурация логгера."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class LoggingSettings(BaseSettings):
    """Класс-конфигурация логгера."""

    silent: bool = Field(
        default=True, description="Отключить вывод логов в консоль (stdout/stderr)"
    )

    logs_path: str = Field(
        default=".log", description="Путь к директории или файлу для сохранения логов", min_length=2
    )

    logs_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="ERROR", description="Уровень логирования для консольного вывода"
    )

    logs_level_file: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Уровень логирования для записи в файл"
    )

    logs_format: str = Field(
        default="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        description="Формат записи логов (использует стандартные форматтеры Python logging)",
        min_length=10,
    )

    maxBytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Максимальный размер файла лога в байтах перед ротацией",
        gt=0,
        le=1024 * 1024 * 1024,  # 1 GB
        alias="max_bytes",
    )

    backupCount: int = Field(
        default=5,
        description="Количество резервных файлов логов для сохранения при ротации",
        ge=1,
        le=25,
        alias="backup_count",
    )
