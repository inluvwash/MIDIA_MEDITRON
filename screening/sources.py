"""Справочник источников, на которые ссылаются правила и рекомендации."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from .schema import ROOT


@lru_cache(maxsize=1)
def all_sources() -> list[dict[str, Any]]:
    """Каталог источников и клинических рекомендаций из knowledge/sources.json."""
    path = ROOT / "knowledge" / "sources.json"
    return json.loads(path.read_text(encoding="utf-8"))


def source_map() -> dict[str, dict[str, Any]]:
    """Источники по их идентификаторам."""
    return {s["id"]: s for s in all_sources()}


def resolve(ids: list[str] | set[str]) -> list[dict[str, Any]]:
    """Превратить идентификаторы в записи источников без дублей, сохраняя порядок."""
    m = source_map()
    out = []
    seen = set()
    for sid in ids:
        if sid in m and sid not in seen:
            out.append(m[sid])
            seen.add(sid)
    return out
