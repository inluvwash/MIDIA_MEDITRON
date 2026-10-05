"""Поиск похожих лабораторных профилей (метрика, подобная расстоянию Говера)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from .casebase import list_cases
from .ml import load_profile
from .schema import ROOT

DATA_PATH = ROOT / "data" / "deficiency_anemia.csv"


@lru_cache(maxsize=1)
def _reference() -> pd.DataFrame:
    """Справочный датасет кейса (840 записей); загружается один раз."""
    return pd.read_csv(DATA_PATH)


def _range(feature: str) -> float:
    """Масштаб признака (p95 - p05) для нормировки расстояния; 1.0, если профиля нет."""
    p = load_profile().get("features", {}).get(feature, {})
    p05, p95 = p.get("p05"), p.get("p95")
    if p05 is None or p95 is None or p95 <= p05:
        return 1.0
    return float(p95 - p05)


def _distance(patient: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, int, int]:
    """Gower-like distance over observed demographic/laboratory overlap only."""
    labs = patient.get("labs", {})
    weighted: list[tuple[float, float]] = []

    if candidate.get("age_years") is not None:
        weighted.append((min(abs(float(patient["age_years"]) - float(candidate["age_years"])) / 60.0, 1.0), 0.35))
    if candidate.get("sex") is not None:
        weighted.append((0.0 if str(patient["sex"]) == str(candidate["sex"]) else 1.0, 0.20))

    lab_overlap = 0
    for feature, value in labs.items():
        if feature not in candidate or candidate.get(feature) is None or pd.isna(candidate.get(feature)):
            continue
        try:
            cv = float(candidate[feature])
        except Exception:
            continue
        weighted.append((min(abs(float(value) - cv) / _range(feature), 1.0), 1.0))
        lab_overlap += 1

    if not weighted:
        return 1.0, 0, 0
    num = sum(d * w for d, w in weighted)
    den = sum(w for _, w in weighted)
    return float(num / den), lab_overlap, len(weighted)


def reference_neighbors(patient: dict[str, Any], k: int = 5, min_lab_overlap: int = 3) -> list[dict[str, Any]]:
    """k ближайших записей справочного датасета; нужно не менее ``min_lab_overlap`` общих анализов."""
    rows: list[dict[str, Any]] = []
    for record in _reference().to_dict("records"):
        dist, lab_overlap, total_overlap = _distance(patient, record)
        if lab_overlap < min_lab_overlap:
            continue
        rows.append({
            "similarity": max(0.0, 1.0 - dist),
            "lab_overlap": lab_overlap,
            "overlap": total_overlap,
            "class_code": record.get("anemia_class"),
            "deficiency_cause": record.get("deficiency_cause"),
            "age_years": record.get("age_years"),
            "sex": record.get("sex"),
            "source": "reference_dataset",
        })
    rows.sort(key=lambda x: (-x["similarity"], -x["lab_overlap"]))
    return rows[:k]


def local_confirmed_neighbors(patient: dict[str, Any], k: int = 5, min_lab_overlap: int = 3) -> list[dict[str, Any]]:
    """Nearest locally confirmed anonymous cases.

    Adding/removing these cases changes retrieval only. It does not retrain the ML model.
    """
    rows: list[dict[str, Any]] = []
    for record in list_cases():
        candidate = {
            "age_years": record.get("age_years"),
            "sex": record.get("sex"),
            **(record.get("labs") or {}),
        }
        dist, lab_overlap, total_overlap = _distance(patient, candidate)
        if lab_overlap < min_lab_overlap:
            continue
        rows.append({
            "case_id": record.get("case_id"),
            "similarity": max(0.0, 1.0 - dist),
            "lab_overlap": lab_overlap,
            "overlap": total_overlap,
            "class_code": record.get("class_code"),
            "confirmed_diagnosis": record.get("confirmed_diagnosis") or record.get("class_label"),
            "icd10": record.get("icd10") or "",
            "age_years": record.get("age_years"),
            "sex": record.get("sex"),
            "source": "local_confirmed",
        })
    rows.sort(key=lambda x: (-x["similarity"], -x["lab_overlap"]))
    return rows[:k]
