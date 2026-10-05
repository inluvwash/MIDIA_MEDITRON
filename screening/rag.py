"""Локальный RAG по разрешённым PDF и опциональное AI-пояснение через Ollama.

LLM получает только уже рассчитанный обезличенный результат и не меняет класс и не назначает лечение.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import requests
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import CLASSES, ROOT
from .sources import all_sources

BOOKS_DIR = ROOT / "books"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

_FORBIDDEN = [
    (r"\bназначаю\b", "рекомендуется обсудить с лечащим врачом"),
    (r"\bназначить\b", "обсудить с лечащим врачом"),
    (r"\bпринимайте\b", "обсудите с лечащим врачом"),
    (r"\bначните принимать\b", "обсудите с лечащим врачом"),
    (r"\bдозировк\w*\b", "схема лечения"),
]


def safety_filter(text: str) -> str:
    out = text or ""
    for pat, repl in _FORBIDDEN:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def ollama_status(timeout: float = 0.8) -> dict[str, Any]:
    """Return server/model readiness without sending patient data."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        r.raise_for_status()
        payload = r.json() if r.content else {}
        names = []
        for item in payload.get("models", []) if isinstance(payload, dict) else []:
            name = str(item.get("name") or item.get("model") or "")
            if name:
                names.append(name)
        target_base = OLLAMA_MODEL.split(":", 1)[0].lower()
        installed = any(
            n.lower() == OLLAMA_MODEL.lower() or n.split(":", 1)[0].lower() == target_base
            for n in names
        )
        return {"server": True, "model": OLLAMA_MODEL, "model_installed": installed, "models": names, "error": None}
    except Exception as exc:
        return {"server": False, "model": OLLAMA_MODEL, "model_installed": False, "models": [], "error": str(exc)}


def ollama_available(timeout: float = 0.35) -> bool:
    status = ollama_status(timeout=timeout)
    return bool(status["server"] and status["model_installed"])


def _chunks_from_text(text: str, source: str, chunk_size: int = 1200, overlap: int = 180) -> list[dict[str, Any]]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append({"source": source, "text": clean[start:end]})
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)
    return chunks


@lru_cache(maxsize=1)
def _knowledge_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    # Structured source registry is always available.
    for s in all_sources():
        chunks.extend(_chunks_from_text((s.get("title", "") + ". " + s.get("note", "")).strip(), s["id"], chunk_size=900, overlap=100))
    # Optional local PDFs. They never leave the machine.
    if BOOKS_DIR.exists():
        for path in sorted(BOOKS_DIR.glob("*.pdf")):
            try:
                reader = PdfReader(str(path))
                for page_no, page in enumerate(reader.pages, 1):
                    text = page.extract_text() or ""
                    for ch in _chunks_from_text(text, f"{path.name}:стр.{page_no}"):
                        chunks.append(ch)
            except Exception:
                continue
    return chunks


def refresh_knowledge_cache() -> None:
    _knowledge_chunks.cache_clear()


def retrieve(query: str, k: int = 6) -> list[dict[str, Any]]:
    chunks = _knowledge_chunks()
    if not chunks or not query.strip():
        return []
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=8000)
    mat = vectorizer.fit_transform(corpus + [query])
    sims = cosine_similarity(mat[-1], mat[:-1])[0]
    idxs = sims.argsort()[::-1][:k]
    out = []
    for i in idxs:
        if sims[i] <= 0:
            continue
        out.append({**chunks[int(i)], "score": float(sims[i])})
    return out


def deterministic_narrative(result: dict[str, Any]) -> str:
    final = result.get("final", {})
    clinical = result.get("clinical", {})
    unc = result.get("uncertainty", {})
    lines = [
        f"Скрининговая гипотеза: {final.get('label', 'не сформирована')}.",
        "Она сформирована из независимых слоёв: проверяемых лабораторных правил, модели на предоставленном датасете и клинического контекста.",
    ]
    if clinical.get("findings"):
        lines.append("Ключевые наблюдения: " + " ".join(f["text"] for f in clinical["findings"][:3]))
    best = unc.get("best_next_measurement")
    if best:
        lines.append(
            f"Главный источник оставшейся неопределённости — конкуренция с гипотезой «{unc.get('alternative', {}).get('label', '')}». "
            f"Для уточнения картины наиболее информативен показатель «{best['label']}»: {best['why']}"
        )
    lines.append("Текст не является диагнозом и не содержит назначения лечения.")
    return "\n\n".join(lines)


def deterministic_patient_narrative(result: dict[str, Any]) -> str:
    final = result.get("final", {})
    unc = result.get("uncertainty", {})
    label = final.get("label") or "однозначная рабочая гипотеза пока не сформирована"
    text = f"Скрининг показывает, что ведущая рабочая версия — «{label}». Это не окончательный диагноз."
    best = unc.get("best_next_measurement")
    if best:
        text += f" Для уточнения картины врач может использовать показатель «{best['label']}», потому что он помогает разделить остающиеся варианты."
    text += " Дальнейшую интерпретацию и план обследования определяет врач с учётом всей клинической картины."
    return text


def _split_generated_text(text: str, result: dict[str, Any]) -> tuple[str, str]:
    """Split one local-LLM response into clinician and patient parts without a second call.

    Models sometimes decorate requested markers with Markdown or replace underscores
    with spaces. Accept those variants so patient-facing text can never leak into
    the clinician section of the PDF.
    """
    raw = str(text or "").strip()

    # Preferred robust sentinels used by the current prompt.
    patient_marker = re.search(r"(?im)(?:[#>*_-]+\s*)?<<<\s*(?:PATIENT|ПАЦИЕНТ)\s*>>>\s*:?\s*", raw)
    doctor_marker = re.search(r"(?im)(?:[#>*_-]+\s*)?<<<\s*(?:DOCTOR|ВРАЧ)\s*>>>\s*:?\s*", raw)

    # Backward-compatible variants from previous prompts / model formatting.
    if patient_marker is None:
        patient_marker = re.search(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*"
            r"(?:ДЛЯ[_\s-]*ПАЦИЕНТА|ПАЦИЕНТУ|ДЛЯ\s+ПАЦИЕНТА)\s*:?\s*"
            r"(?:\*\*|__)?\s*:?\s*$",
            raw,
        )

    if patient_marker:
        doctor = raw[: patient_marker.start()].strip()
        patient = raw[patient_marker.end() :].strip()
    else:
        doctor = raw
        patient = deterministic_patient_narrative(result)

    # Strip doctor marker if present.
    doctor = re.sub(r"(?im)\s*(?:[#>*_-]+\s*)?<<<\s*(?:DOCTOR|ВРАЧ)\s*>>>\s*:?\s*", "", doctor).strip()
    doctor = re.sub(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*ДЛЯ[_\s-]*ВРАЧА\s*:?\s*(?:\*\*|__)?\s*:?\s*",
        "",
        doctor,
        count=1,
    ).strip()
    patient = re.sub(r"(?im)\s*(?:[#>*_-]+\s*)?<<<\s*(?:PATIENT|ПАЦИЕНТ)\s*>>>\s*:?\s*", "", patient).strip()
    patient = re.sub(
        r"(?im)^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:ДЛЯ[_\s-]*ПАЦИЕНТА|ПАЦИЕНТУ)\s*:?\s*(?:\*\*|__)?\s*:?\s*",
        "",
        patient,
        count=1,
    ).strip()

    return doctor or deterministic_narrative(result), patient or deterministic_patient_narrative(result)




def _mostly_russian(text: str) -> bool:
    value = str(text or "")
    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", value)
    if not letters:
        return False
    cyr = re.findall(r"[А-Яа-яЁё]", value)
    # Require overwhelmingly Russian prose. Short conventional lab abbreviations
    # may remain Latin, but English words/sentences force the safe Russian fallback.
    latin_words = re.findall(r"\b[A-Za-z]{3,}\b", value)
    allowed = {"MCV", "MCH", "MCHC", "RDW", "TSAT", "CRP", "B12", "TIBC", "UIBC"}
    foreign_words = [w for w in latin_words if w.upper() not in allowed]
    return len(cyr) / max(1, len(letters)) >= 0.88 and len(foreign_words) == 0


def _limit_sentences(text: str, maximum: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return value
    parts = re.split(r"(?<=[.!?])\s+", value)
    return " ".join(parts[:maximum]).strip()


def generate_ai_explanation(
    result: dict[str, Any],
    patient: dict[str, Any],
    progress: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    def emit(percent: int, message: str) -> None:
        if progress is not None:
            progress(int(percent), message)

    emit(5, "Подготавливаем структурированный результат")
    query = " ".join([
        result.get("final", {}).get("label", ""),
        result.get("uncertainty", {}).get("alternative", {}).get("label", ""),
        patient.get("context", {}).get("complaints", ""),
        patient.get("context", {}).get("history", ""),
    ])
    emit(15, "Подбираем релевантные источники")
    snippets = retrieve(query, k=6)
    emit(25, "Источники подготовлены")
    status = ollama_status(timeout=0.8)
    if not status["server"]:
        emit(100, "Готово: использован детерминированный режим")
        return {
            "mode": "deterministic",
            "text": deterministic_narrative(result),
            "patient_text": deterministic_patient_narrative(result),
            "sources": snippets,
            "warning": "Ollama не запущена: использован детерминированный текст без генеративной модели.",
        }
    if not status["model_installed"]:
        emit(100, "Готово: целевая модель Ollama не найдена")
        return {
            "mode": "deterministic",
            "text": deterministic_narrative(result),
            "patient_text": deterministic_patient_narrative(result),
            "sources": snippets,
            "warning": f"Ollama запущена, но модель {OLLAMA_MODEL} не установлена. Использован детерминированный текст.",
        }

    emit(35, "Готовим обезличенный контекст для локальной модели")
    compact = {
        "final": result.get("final"),
        "clinical_findings": result.get("clinical", {}).get("findings", []),
        "alerts": result.get("clinical", {}).get("alerts", []),
        "ml_top": list((result.get("model") or {}).get("class_scores", {}).items())[:4],
        "uncertainty": result.get("uncertainty"),
        "causal_chains": result.get("causal_chains", []),
        "context": patient.get("context", {}),
    }
    source_text = "\n\n".join(f"[{s['source']}] {s['text']}" for s in snippets)
    system = (
        "Ты медицинский редактор локальной СППР. Отвечай ТОЛЬКО на русском языке, кириллицей. "
        "Используй исключительно уже рассчитанные данные предыдущих слоёв и приложенные источники. "
        "МАТЕМАТИКУ НЕ ПЕРЕСЧИТЫВАЙ, не меняй class_code, scores и лабораторные значения. "
        "Не ставь окончательный диагноз, не назначай лекарства, дозировки, курсы, БАД, диету или изменение образа жизни. "
        "Не добавляй причин, которых нет в causal/context слоях. "
        "Ответ раздели строго двумя маркерами на отдельных строках: <<<ВРАЧ>>> и <<<ПАЦИЕНТ>>>. "
        "После <<<ВРАЧ>>> дай 3–6 коротких предложений: ведущая гипотеза, главные поддерживающие факты, оставшаяся неопределённость и один следующий информативный показатель, если он рассчитан. "
        "После <<<ПАЦИЕНТ>>> дай ровно 2–3 коротких предложения простым русским языком: что показал скрининг, что ещё требует уточнения и что окончательную оценку делает врач. "
        "Никаких назначений, эмоциональных обещаний и иностранного текста. Не используй Markdown-заголовки с #. "
        "Если данных мало — прямо скажи это."
    )
    user = "СТРУКТУРИРОВАННЫЙ РЕЗУЛЬТАТ:\n" + json.dumps(compact, ensure_ascii=False, indent=2) + "\n\nИСТОЧНИКИ/RAG:\n" + source_text
    try:
        emit(45, "Запускаем локальную генерацию")
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": True,
                "keep_alive": "10m",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            stream=True,
            timeout=(5, 180),
        )
        r.raise_for_status()
        pieces: list[str] = []
        chunk_count = 0
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            token = str((payload.get("message") or {}).get("content") or "")
            if token:
                pieces.append(token)
                chunk_count += 1
                pct = min(94, 50 + chunk_count // 2)
                emit(pct, f"Идёт генерация: получено фрагментов — {chunk_count}")
            if payload.get("done"):
                break
        emit(96, "Проверяем текст safety-фильтром")
        generated = safety_filter("".join(pieces).strip())
        if not generated:
            raise RuntimeError("Пустой ответ Ollama")
        doctor_text, patient_text = _split_generated_text(generated, result)
        if not _mostly_russian(doctor_text) or not _mostly_russian(patient_text):
            doctor_text = deterministic_narrative(result)
            patient_text = deterministic_patient_narrative(result)
            mode = "deterministic"
            warning = "Ответ локальной модели был не на русском языке; использован безопасный русскоязычный резервный текст."
        else:
            doctor_text = _limit_sentences(doctor_text, 6)
            patient_text = _limit_sentences(patient_text, 3)
            mode = "ollama"
            warning = None
        emit(100, "Объяснение готово")
        return {
            "mode": mode,
            "text": doctor_text,
            "patient_text": patient_text,
            "sources": snippets,
            "warning": warning,
        }
    except Exception as exc:
        emit(100, "Готово: использован резервный детерминированный режим")
        return {
            "mode": "deterministic",
            "text": deterministic_narrative(result),
            "patient_text": deterministic_patient_narrative(result),
            "sources": snippets,
            "warning": f"Генеративный слой недоступен ({exc}); использован детерминированный текст.",
        }
