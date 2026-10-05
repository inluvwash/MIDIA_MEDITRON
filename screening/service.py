"""Оркестрация слоёв скрининга: единая функция ``screen`` для интерфейса и API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .causal import build_chains
from .coding import coding_candidates
from .communication import build_doctor_brief, build_patient_conversation
from .clinical import evaluate
from .ml import ModelUnavailable, predict
from .schema import CLASSES, normalize_record
from .similar import local_confirmed_neighbors, reference_neighbors
from .sources import resolve
from .uncertainty import build_uncertainty_map


def _confidence_band(model: dict[str, Any] | None, clinical: dict[str, Any], final_code: str | None) -> str:
    """Словесная оценка согласованности слоёв (не вероятность диагноза).

    «Высокая»: сильный скоринг (>= 0.75), заметный отрыв от второго класса (>= 0.30)
    и совпадение с клиническими правилами. «Умеренная»: скоринг >= 0.55, отрыв >= 0.15.
    Иначе — «повышенная неопределённость». Пороги рабочие, не клинически калиброванные.
    """
    if not final_code:
        return "недостаточно данных"
    if not model:
        return "клинические правила без ML"
    top = float(model.get("top_score", 0))
    margin = float(model.get("margin", 0))
    agree = clinical.get("rule_class") == final_code
    if top >= 0.75 and margin >= 0.30 and agree:
        return "высокая согласованность слоёв"
    if top >= 0.55 and margin >= 0.15:
        return "умеренная согласованность"
    return "повышенная неопределённость"


def _select_final(clinical: dict[str, Any], model: dict[str, Any] | None) -> tuple[str | None, str]:
    """Выбрать итоговый класс и указать, чем он обоснован.

    ML-класс приоритетен; расхождение с клиническими правилами не скрывается, а
    помечается методом ``ml_with_rule_disagreement``. Без модели используются правила.
    """
    rule = clinical.get("rule_class")
    if model and model.get("top_class"):
        top = model["top_class"]
        if rule and rule != top:
            return top, "ml_with_rule_disagreement"
        return top, "ml_and_rules" if rule == top else "ml_primary"
    return rule, "rules_only" if rule else "insufficient"


def _coexisting_signals(clinical: dict[str, Any], final_code: str | None) -> list[dict[str, str]]:
    """Signals that may coexist with the leading class.

    This is deliberately separate from the uncertainty map: an alternative hypothesis
    competes for the same explanation, while a coexisting signal may be present at the
    same time. The function does not create new diagnoses.
    """
    deficit_labels = {
        "iron": "Признаки дефицита / ограничения доступности железа",
        "b12": "Признаки B12-дефицитного направления",
        "folate": "Признаки фолат-дефицитного направления",
        "b6": "Признаки B6-дефицитного направления",
        "copper": "Признаки дефицита меди",
    }
    primary_signal = {
        "latent_deficiency": "iron",
        "iron_deficiency_anemia": "iron",
        "B12_deficiency_anemia": "b12",
        "B12_deficiency_no_anemia": "b12",
        "folate_deficiency_anemia": "folate",
        "folate_deficiency_no_anemia": "folate",
        "B6_deficiency": "b6",
        "copper_deficiency": "copper",
    }.get(final_code)

    out: list[dict[str, str]] = []
    for key in clinical.get("candidate_deficits", []):
        if key not in deficit_labels:
            continue
        if final_code != "mixed_deficiency" and key == primary_signal:
            continue
        out.append({"type": "deficiency_signal", "label": deficit_labels[key]})

    for item in clinical.get("exclusions", []):
        if item.get("status") == "abnormal":
            out.append({
                "type": "parallel_direction",
                "label": f"Параллельное направление: {item.get('title', 'дополнительная причина')}",
            })

    return out


def screen(
    raw: dict[str, Any],
    units_confirmed: bool = False,
    include_neighbors: bool = True,
    include_uncertainty: bool = True,
    include_explanations: bool = True,
) -> dict[str, Any]:
    """Полный скрининг одного пациента — единая точка входа для UI и API.

    Слои результата по порядку: нормализация входа -> клинические правила -> ML ->
    итоговый класс -> карта неопределённости -> причинные цепочки -> похожие профили ->
    справочные коды МКБ-10 -> источники -> тексты для врача и пациента.
    ФИО пациента сюда не передаётся и в результат не попадает.
    """
    # 1. Нормализация: единицы приводятся к каноническим, проверяются возраст и пол.
    patient = normalize_record(raw, units_confirmed=units_confirmed)
    # 2. Прозрачные клинические правила (порог Hb по ВОЗ, признаки дефицитов, исключения).
    clinical = evaluate(patient)
    # 3. ML-слой; при отсутствии модели сервис деградирует до одних правил.
    try:
        model = predict(patient, explain=include_explanations)
    except ModelUnavailable:
        model = None

    # 4. Итоговый класс и 5. карта неопределённости (альтернативная гипотеза, след. маркер).
    final_code, method = _select_final(clinical, model)
    uncertainty = (
        build_uncertainty_map(patient, model)
        if include_uncertainty
        else {"status": "skipped", "message": "Карта неопределённости пропущена.", "items": []}
    )

    # 6. Причинные цепочки строятся для ведущей и (если есть) альтернативной гипотезы.
    class_codes_for_chain: list[str] = []
    if final_code:
        class_codes_for_chain.append(final_code)
    alt = uncertainty.get("alternative", {}).get("code") if isinstance(uncertainty, dict) else None
    if alt and alt not in class_codes_for_chain:
        class_codes_for_chain.append(alt)
    causal = build_chains(patient, class_codes_for_chain[:2])

    # 7. Похожие профили: только для сопоставления, не как доказательство диагноза.
    references = reference_neighbors(patient, k=5) if include_neighbors else []
    local_references = local_confirmed_neighbors(patient, k=5) if include_neighbors else []
    coding = coding_candidates(final_code)

    # 8. Сбор источников, на которые опираются сработавшие правила и рекомендации.
    source_ids: list[str] = ["case_su", "dataset_su", "variables_su", "who_hb_2024", "minzdrav_cr_registry_2026"]
    for f in clinical.get("findings", []):
        source_ids.extend(f.get("source_ids", []))
    for r in clinical.get("recommendations", []):
        source_ids.extend(r.get("source_ids", []))
    for item in uncertainty.get("items", []) if isinstance(uncertainty, dict) else []:
        source_ids.extend(item.get("source_ids", []))
    source_ids.extend(coding.get("source_ids", []))

    result = {
        "schema_version": 8,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok" if final_code else "incomplete",
        "patient": patient,
        "clinical": clinical,
        "model": model,
        "final": {
            "class_code": final_code,
            "label": CLASSES.get(final_code, "Недостаточно данных для класса") if final_code else "Недостаточно данных для класса",
            "method": method,
            "confidence_band": _confidence_band(model, clinical, final_code),
            "rule_class": clinical.get("rule_class"),
            "model_class": model.get("top_class") if model else None,
            "agreement": bool(model and clinical.get("rule_class") and model.get("top_class") == clinical.get("rule_class")),
        },
        "coexisting_signals": _coexisting_signals(clinical, final_code),
        "uncertainty": uncertainty,
        "causal_chains": causal,
        "coding": coding,
        "similar_cases": {
            "local_confirmed": local_references,
            "reference_dataset": references,
            "note": "Локально подтверждённые случаи и reference-выборка служат только для поиска похожих профилей и не являются самостоятельным доказательством диагноза. Локальная база не переобучает ML автоматически.",
        },
        "sources": resolve(source_ids),
        "warnings": list(dict.fromkeys(patient.get("warnings", []))),
    }

    if method == "ml_with_rule_disagreement":
        result["warnings"].append(
            "Клинические правила и ML-слой расходятся. Итоговый class_code взят из модели, но несогласие показано явно и должно рассматриваться как дополнительная неопределённость."
        )
    result["doctor_brief"] = build_doctor_brief(result)
    result["patient_conversation"] = build_patient_conversation(result)
    return result
