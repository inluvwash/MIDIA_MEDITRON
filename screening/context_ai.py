"""Структурирование клинического контекста из документа локальной LLM с маскировкой идентификаторов."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import requests
from pypdf import PdfReader

from .rag import OLLAMA_MODEL, OLLAMA_URL, ollama_status


def extract_document_text(data: bytes, filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if data.lstrip().startswith(b"%PDF") or suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return ""


def _mask_identifiers(text: str) -> str:
    value = str(text or "")
    value = re.sub(
        r"(?im)^\s*(?:фио(?:\s+пациента)?|ф\.?\s*и\.?\s*о\.?(?:\s+пациента)?|patient\s+name)\s*[:=—-]\s*.*$",
        "ФИО: [СКРЫТО]",
        value,
    )
    value = re.sub(r"(?im)^\s*(?:телефон|e-?mail|снилс|полис|адрес)\s*[:=—-]\s*.*$", "", value)
    return value[:18000]


def _parse_json_block(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Нейросеть не вернула JSON")
    payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Некорректный ответ нейросети")
    return payload


def structure_clinical_context(data: bytes, filename: str, timeout: float = 45.0) -> dict[str, Any]:
    """Use a *local* Ollama model only to structure free clinical prose.

    The model is not allowed to calculate laboratory values, infer a diagnosis or
    invent absent details.  FIO and obvious direct identifiers are masked before the
    request.  The caller can always ignore this result and use the deterministic parser.
    """
    status = ollama_status(timeout=0.8)
    if not (status.get("server") and status.get("model_installed")):
        return {"ok": False, "warning": "Локальная нейросеть недоступна — использован обычный парсер.", "data": {}}

    text = _mask_identifiers(extract_document_text(data, filename))
    if len(text.strip()) < 20:
        return {"ok": False, "warning": "В документе недостаточно текста для ИИ-разбора.", "data": {}}

    prompt = f"""
Ты локальный помощник по СТРУКТУРИРОВАНИЮ медицинского документа. Не ставь диагноз и не давай медицинских рекомендаций.
Нужно только разложить ЯВНО написанный текст по трем полям: complaints, history, medications.
Не додумывай факты. Не переносись значения лабораторных анализов в эти поля. Не включай административный текст.
Если поле отсутствует, верни пустую строку. Ответ ТОЛЬКО на русском языке и ТОЛЬКО валидным JSON без Markdown:
{{"complaints":"...","history":"...","medications":"..."}}

Документ:
{text}
""".strip()
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json() if response.content else {}
        parsed = _parse_json_block(payload.get("response", ""))
        out = {}
        for key in ("complaints", "history", "medications"):
            value = re.sub(r"\s+", " ", str(parsed.get(key, "") or "")).strip()
            # The clinical-text helper is Russian-only. If the model switches to
            # English prose, ignore its output and keep the deterministic parser.
            latin_words = re.findall(r"\b[A-Za-z]{3,}\b", value)
            if latin_words:
                return {"ok": False, "warning": "ИИ-разбор вернул текст не на русском языке — использован обычный парсер.", "data": {}}
            out[key] = value[:1600]
        return {"ok": True, "warning": None, "data": out}
    except Exception as exc:
        return {"ok": False, "warning": f"ИИ-разбор не завершён ({exc}). Использован обычный парсер.", "data": {}}
