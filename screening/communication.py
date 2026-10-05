"""Тексты результата для двух аудиторий: краткая сводка врача и разговор с пациентом."""

from __future__ import annotations

from typing import Any

from .schema import CLASSES, LABS


def _short_lab(patient: dict[str, Any], marker: str) -> str | None:
    if marker not in patient.get("labs", {}):
        return None
    value = patient["labs"][marker]
    unit = patient.get("units", {}).get(marker, LABS.get(marker, {}).get("unit", ""))
    return f"{LABS.get(marker, {}).get('label', marker)} {value:g} {unit}".strip()


def _lab_visual(result: dict[str, Any], marker: str) -> dict[str, Any] | None:
    patient = result.get("patient", {})
    labs = patient.get("labs", {})
    if marker not in labs:
        return None
    value = float(labs[marker])
    unit = patient.get("units", {}).get(marker, LABS.get(marker, {}).get("unit", ""))
    label = LABS.get(marker, {}).get("label", marker)
    clinical = result.get("clinical", {})
    status = "в пределах рабочего ориентира"
    direction = "•"
    level = "neutral"

    if marker == "hemoglobin":
        threshold = float(clinical.get("threshold") or 0)
        if threshold and value < threshold:
            ratio = value / threshold
            direction, status, level = ("↓↓", "выраженно снижен", "strong_low") if ratio < 0.8 else ("↓", "снижен", "low")
    elif marker == "ferritin":
        if value < 15:
            direction, status, level = "↓↓", "выраженно снижен", "strong_low"
        elif value < 30:
            direction, status, level = "↓", "снижен", "low"
    elif marker == "TSAT":
        if value < 10:
            direction, status, level = "↓↓", "выраженно снижено", "strong_low"
        elif value < 20:
            direction, status, level = "↓", "снижено", "low"
    elif marker == "Ret_He" and value < 28:
        direction, status, level = "↓", "снижен", "low"
    elif marker == "MCV":
        if value < 80:
            direction, status, level = "↓", "снижен", "low"
        elif value > 100:
            direction, status, level = "↑", "повышен", "high"
    elif marker == "RDW" and value > 15:
        direction, status, level = "↑", "повышен", "high"
    elif marker == "CRP" and value > 5:
        direction, status, level = "↑", "повышен", "high"
    elif marker == "vitamin_B12":
        if value < 180:
            direction, status, level = "↓↓", "снижен", "strong_low"
        elif value <= 350:
            direction, status, level = "≈", "пограничная зона", "borderline"
    elif marker == "folate":
        if value < 3:
            direction, status, level = "↓↓", "снижен", "strong_low"
        elif value < 4:
            direction, status, level = "↓", "погранично снижен", "low"

    return {
        "feature": marker,
        "label": label,
        "value": value,
        "unit": unit,
        "direction": direction,
        "status": status,
        "level": level,
        "text": f"{label}: {value:g} {unit} — {direction} {status}".strip(),
    }


def _patient_feels(patient: dict[str, Any], code: str) -> list[str]:
    context = patient.get("context", {})
    symptoms = [str(x) for x in context.get("symptoms", []) if str(x).strip()]
    complaints = str(context.get("complaints", "") or "").lower().replace("ё", "е")
    associations = {
        "iron_deficiency_anemia": ["Утомляемость / слабость", "Снижение переносимости физической нагрузки", "Головокружение", "Одышка при нагрузке", "Выпадение волос / ломкость ногтей", "Синдром беспокойных ног"],
        "latent_deficiency": ["Утомляемость / слабость", "Выпадение волос / ломкость ногтей", "Снижение переносимости физической нагрузки"],
        "B12_deficiency_anemia": ["Утомляемость / слабость", "Парестезии / покалывание", "Онемение / снижение чувствительности", "Нарушение походки / равновесия", "Снижение концентрации / памяти"],
        "B12_deficiency_no_anemia": ["Парестезии / покалывание", "Онемение / снижение чувствительности", "Снижение концентрации / памяти"],
        "folate_deficiency_anemia": ["Утомляемость / слабость", "Глоссит / жжение языка", "Стоматит / изменения слизистой полости рта"],
        "inflammation_anemia": ["Утомляемость / слабость", "Снижение переносимости физической нагрузки", "Длительная лихорадка / субфебрилитет", "Суставные боли / признаки хронического воспаления"],
        "mixed_deficiency": symptoms,
    }
    allowed = associations.get(code, symptoms)
    chosen = [s for s in symptoms if s in allowed]
    if chosen:
        return chosen[:4]
    # Preserve literal complaint phrases only as a last-resort patient-facing clue.
    simple_terms = []
    for needle, label in [
        ("слаб", "слабость"), ("утом", "утомляемость"), ("головокруж", "головокружение"),
        ("одыш", "одышка"), ("волос", "выпадение волос"), ("ногт", "ломкость ногтей"),
        ("парест", "покалывание / парестезии"), ("онемен", "онемение"),
    ]:
        if needle in complaints:
            simple_terms.append(label)
    return simple_terms[:4]


def build_doctor_brief(result: dict[str, Any]) -> dict[str, Any]:
    clinical = result.get("clinical", {})
    final = result.get("final", {})
    unc = result.get("uncertainty", {})
    patient = result.get("patient", {})

    priority: list[str] = []
    for alert in clinical.get("alerts", [])[:2]:
        priority.append(alert)
    if final.get("label"):
        priority.append(f"Ведущая скрининговая гипотеза: {final['label']}.")
    if final.get("rule_class") and final.get("model_class") and not final.get("agreement"):
        priority.append("Клинические правила и ML-слой расходятся — конфликт показан явно.")
    best = unc.get("best_next_measurement")
    if best:
        priority.append(f"Наиболее информативное недостающее измерение: {best['label']}.")
    if not priority:
        priority.append("Данных недостаточно для устойчивой гипотезы; сначала требуется проверить полноту ключевых показателей.")

    keys = ("hemoglobin", "ferritin", "TSAT", "Ret_He", "vitamin_B12", "folate", "CRP")
    anchors = [x for x in (_short_lab(patient, key) for key in keys) if x]
    anchors_visual = [x for x in (_lab_visual(result, key) for key in keys) if x]

    return {
        "headline": final.get("label") or "Недостаточно данных для класса",
        "priority": priority[:4],
        "anchors": anchors[:6],
        "anchors_visual": anchors_visual[:6],
        "uncertainty_status": unc.get("status", "unavailable"),
    }


def build_patient_conversation(result: dict[str, Any]) -> dict[str, Any]:
    final = result.get("final", {})
    unc = result.get("uncertainty", {})
    patient = result.get("patient", {})
    model = result.get("model") or {}

    summary = (
        f"Ведущая рабочая гипотеза — «{final['label']}». Окончательный диагноз ставит врач."
        if final.get("label")
        else "Данных пока недостаточно для устойчивой рабочей гипотезы. Окончательную оценку выполняет врач."
    )

    keys = ("hemoglobin", "ferritin", "TSAT", "Ret_He", "vitamin_B12", "folate", "CRP")
    facts_visual = [x for x in (_lab_visual(result, key) for key in keys) if x][:4]
    facts = [x["text"] for x in facts_visual]

    top = unc.get("top_hypothesis") or {}
    alt = unc.get("alternative") or {}
    if top and alt:
        uncertainty_plain = (
            f"Сейчас наиболее близки две версии: «{top.get('label', '')}» и «{alt.get('label', '')}». "
            "Они могут конкурировать или частично сочетаться."
        )
    else:
        uncertainty_plain = unc.get("message") or "Выраженная конкурирующая гипотеза сейчас не выделена."

    best = unc.get("best_next_measurement")
    if best and top and alt:
        next_step = f"Анализ «{best['label']}» поможет лучше различить «{top.get('label', '')}» и «{alt.get('label', '')}» и уточнить картину."
    elif best:
        next_step = f"Анализ «{best['label']}» поможет уменьшить оставшуюся неопределённость и уточнить картину."
    else:
        next_step = "Необходимость дальнейшего обследования определяет врач после сопоставления результата с клинической картиной."

    ranked: list[dict[str, Any]] = []
    for code, score in list(model.get("class_scores", {}).items())[:3]:
        ranked.append({
            "code": code,
            "label": CLASSES.get(code, code),
            "rating": max(0, min(100, int(round(float(score) * 100)))),
        })

    causal_simple: list[dict[str, Any]] = []
    for chain in result.get("causal_chains", [])[:2]:
        code = str(chain.get("class_code") or "")
        causal_simple.append({
            "observed": "; ".join(chain.get("markers", [])[:3]) or "Лабораторные данные",
            "observed_items": chain.get("markers", [])[:3],
            "hypothesis": chain.get("hypothesis", chain.get("label", "")),
            "patient_feels": _patient_feels(patient, code),
            "possible_context": "; ".join(chain.get("causes", [])[:2]) or "Причина требует клинического уточнения",
        })

    return {
        "plain_summary": summary,
        "key_facts": facts,
        "key_facts_visual": facts_visual,
        "uncertainty_plain": uncertainty_plain,
        "why_next_step": next_step,
        "model_ranking": ranked,
        "causal_simple": causal_simple,
        "source_ids": ["case_su"],
    }
