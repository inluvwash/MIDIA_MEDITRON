"""Локальная база подтверждённых анонимных случаев.

Это НЕ продольная история пациента и НЕ автоматическое переобучение ML.
База используется только как дополнительный источник похожих подтверждённых
лабораторных профилей. ФИО и свободный клинический текст здесь не хранятся.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .schema import CLASSES, ROOT

CASE_PATH = Path(os.getenv("LOCAL_CASES_PATH", str(ROOT / "data" / "confirmed_cases.json")))
MAX_CASES = 5000


def _read() -> list[dict[str, Any]]:
    if not CASE_PATH.exists():
        return []
    try:
        data = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _write(rows: list[dict[str, Any]]) -> None:
    CASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CASE_PATH.with_suffix(CASE_PATH.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CASE_PATH)


def list_cases() -> list[dict[str, Any]]:
    return _read()


def add_case(
    patient: dict[str, Any],
    class_code: str,
    confirmed_diagnosis: str = "",
    icd10: str = "",
) -> dict[str, Any]:
    if class_code not in CLASSES:
        raise ValueError("Выберите один из 12 скрининговых классов")
    rows = _read()
    if len(rows) >= MAX_CASES:
        raise ValueError(f"Локальная база достигла лимита {MAX_CASES} случаев")

    # Намеренно не сохраняем patient_id/FIO, жалобы, анамнез или препараты.
    row = {
        "case_id": uuid4().hex[:12],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "age_years": float(patient["age_years"]),
        "sex": str(patient["sex"]),
        "labs": {str(k): float(v) for k, v in patient.get("labs", {}).items()},
        "units": {str(k): str(v) for k, v in patient.get("units", {}).items()},
        "class_code": class_code,
        "class_label": CLASSES[class_code],
        "confirmed_diagnosis": str(confirmed_diagnosis or "").strip(),
        "icd10": str(icd10 or "").strip(),
    }
    rows.append(row)
    _write(rows)
    return row


def delete_case(case_id: str) -> bool:
    rows = _read()
    kept = [r for r in rows if str(r.get("case_id")) != str(case_id)]
    if len(kept) == len(rows):
        return False
    _write(kept)
    return True
