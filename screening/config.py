"""Загрузка конфигурации из ``config/.env``.

Параметры сервиса (адрес локальной LLM, API-ключ, путь к локальной базе случаев)
вынесены из кода в переменные окружения. Для локального запуска без Docker их
можно задать файлом ``config/.env`` (шаблон — ``config/.env.example``).

Правила:
* файл необязателен — без него действуют значения по умолчанию из кода;
* уже заданные переменные окружения НЕ перезаписываются (приоритет у окружения,
  в том числе у переменных, которые передаёт docker-compose);
* внешние зависимости не нужны: формат ``KEY=VALUE``, строки с ``#`` — комментарии.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / "config" / ".env"


def load_env(path: Path = ENV_PATH) -> None:
    """Прочитать ``path`` и добавить отсутствующие переменные в ``os.environ``."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        # Пустые значения (например, HEALTHSCREENING_API_KEY=) не задаём:
        # пустой ключ означает «защита выключена» и совпадает с поведением по умолчанию.
        if key and value and key not in os.environ:
            os.environ[key] = value


load_env()
