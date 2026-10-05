"""ML-слой: 12-классовая модель RandomForest и вспомогательные расчёты.

Модель обучена на предоставленном датасете (``train.py``). Здесь выполняется только
вывод: предсказание, согласование с критерием анемии ВОЗ и оценка чувствительности.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .schema import CLASS_ORDER, FEATURES, ROOT, anemia_state, class_is_consistent, feature_row

MODEL_PATH = ROOT / "models" / "anemia_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
PROFILE_PATH = ROOT / "models" / "data_profile.json"


class ModelUnavailable(RuntimeError):
    """Модель не найдена на диске; сервис продолжает работать на клинических правилах."""


@lru_cache(maxsize=1)
def load_bundle() -> tuple[dict[str, Any], dict[str, Any]]:
    """Загрузить обученную модель и её метаданные (кэшируется на время работы процесса)."""
    if not MODEL_PATH.exists():
        raise ModelUnavailable("Обученная модель не найдена. Запустите python train.py")
    bundle = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    return bundle, meta


@lru_cache(maxsize=1)
def load_profile() -> dict[str, Any]:
    """Профиль обучающей выборки: перцентили признаков, медианы и покрытие по классам.

    Используется для оценки чувствительности и карты неопределённости.
    """
    if not PROFILE_PATH.exists():
        return {}
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _frame(patient: dict[str, Any]) -> pd.DataFrame:
    """Одна строка признаков пациента в порядке, на котором обучалась модель."""
    return pd.DataFrame([feature_row(patient)], columns=FEATURES)


def _raw_scores_many(patients: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Predict class scores for several patient variants in one sklearn call.

    This keeps the exact same model and post-processing logic while avoiding dozens
    of repeated RandomForest predict_proba calls in sensitivity/uncertainty analysis.
    """
    if not patients:
        return []
    bundle, _ = load_bundle()
    model = bundle["model"]
    frame = pd.DataFrame([feature_row(p) for p in patients], columns=FEATURES)
    proba = model.predict_proba(frame)
    classes = [str(c) for c in model.classes_]
    return [{c: float(v) for c, v in zip(classes, row)} for row in proba]


def _raw_scores(patient: dict[str, Any]) -> dict[str, float]:
    """Сырые скоринги RandomForest по 12 классам без клинических ограничений."""
    return _raw_scores_many([patient])[0]


def _constrain_one(patient: dict[str, Any], scores: dict[str, float]) -> dict[str, float]:
    """Согласовать скоринги модели с гемоглобиновым критерием ВОЗ.

    Классы «…_anemia» невозможны при Hb выше порога (Ж < 120, М < 130 г/л), а классы
    «…_no_anemia» — при Hb ниже порога. Несовместимые классы убираются, остальные
    скоринги перенормируются на сумму 1. Так модель не может выдать анемию
    у пациента с нормальным гемоглобином (и наоборот).
    """
    anemia, _ = anemia_state(patient)
    kept = {c: p for c, p in scores.items() if class_is_consistent(c, anemia)}
    total = sum(kept.values())
    if total <= 0:
        # Модель не дала вес ни одному допустимому классу — оставляем сырые скоринги.
        return scores
    return {c: p / total for c, p in kept.items()}


def constrained_scores(patient: dict[str, Any]) -> dict[str, float]:
    """Скоринги классов после согласования с критерием анемии (см. ``_constrain_one``)."""
    return _constrain_one(patient, _raw_scores(patient))


def constrained_scores_many(patients: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Пакетный вариант ``constrained_scores`` для нескольких вариантов пациента."""
    raw_rows = _raw_scores_many(patients)
    return [_constrain_one(patient, scores) for patient, scores in zip(patients, raw_rows)]


def scores_with_overrides(
    patient: dict[str, Any], overrides: list[tuple[str, float]]
) -> list[dict[str, float]]:
    """Batch-score patient variants where one laboratory feature is overridden."""
    variants: list[dict[str, Any]] = []
    for feature, value in overrides:
        modified = {**patient, "labs": dict(patient["labs"])}
        modified["labs"][feature] = float(value)
        variants.append(modified)
    return constrained_scores_many(variants)


def _local_sensitivity(patient: dict[str, Any], top_class: str, max_features: int = 10) -> list[dict[str, Any]]:
    """Model sensitivity to replacing one observed value by the training median.

    The variants are scored in one batch for speed. This is intentionally described
    as sensitivity, not causality or SHAP.
    """
    profile = load_profile()
    medians = profile.get("features", {})
    labs = patient["labs"]
    candidates: list[tuple[str, float, float]] = []
    overrides: list[tuple[str, float]] = []
    for feature, value in labs.items():
        if feature not in FEATURES or feature in {"age_years", "sex"}:
            continue
        med = medians.get(feature, {}).get("p50")
        if med is None:
            continue
        candidates.append((feature, float(value), float(med)))
        overrides.append((feature, float(med)))

    if not overrides:
        return []
    base = constrained_scores(patient).get(top_class, 0.0)
    variant_scores = scores_with_overrides(patient, overrides)
    rows: list[dict[str, Any]] = []
    for (feature, value, med), scores in zip(candidates, variant_scores):
        new_score = scores.get(top_class, 0.0)
        rows.append({
            "feature": feature,
            "value": value,
            "median": med,
            "score_change": float(base - new_score),
        })
    rows.sort(key=lambda x: abs(x["score_change"]), reverse=True)
    return rows[:max_features]


def predict(patient: dict[str, Any], explain: bool = True) -> dict[str, Any]:
    """ML-слой: ранжирование 12 классов для одного пациента.

    Возвращает отсортированные скоринги, ведущий класс, отрыв от второго класса
    (``margin`` — основа оценки неопределённости) и чувствительность к признакам.
    Скоринги — относительный рейтинг, а не клинически откалиброванная вероятность.
    """
    bundle, meta = load_bundle()
    scores = constrained_scores(patient)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_class, top_score = ranked[0]
    margin = top_score - (ranked[1][1] if len(ranked) > 1 else 0.0)
    return {
        "model_id": meta.get("model_id"),
        "class_scores": dict(ranked),
        "top_class": top_class,
        "top_score": float(top_score),
        "margin": float(margin),
        "score_note": meta.get("probability_note", "Модельные scores не являются клинически откалиброванными вероятностями."),
        "sensitivity": _local_sensitivity(patient, top_class) if explain else [],
    }


def score_with_override(patient: dict[str, Any], feature: str, value: float) -> dict[str, float]:
    """Скоринги при подстановке одного значения признака (контрфактический сценарий)."""
    return scores_with_overrides(patient, [(feature, value)])[0]
