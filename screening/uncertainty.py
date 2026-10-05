"""Карта неопределённости: альтернативная гипотеза и следующий наиболее информативный маркер.

Это инструмент информационной ценности, а не автоматическое назначение исследований.
"""

from __future__ import annotations

import math
from typing import Any

from .ml import load_profile, scores_with_overrides
from .schema import CLASSES, LABS

# Экспертные приоритеты дополнительных маркеров для типичных пар конкурирующих
# гипотез: (признак, клиническое обоснование, id источников из knowledge/sources.json).
# Порядок в списке = убывающая важность. Для остальных пар используется data-driven оценка.
PAIR_PRIORITY: dict[frozenset[str], list[tuple[str, str, list[str]]]] = {
    frozenset({"iron_deficiency_anemia", "inflammation_anemia"}): [
        ("Ret_He", "Ret-He отражает доступность железа для текущего эритропоэза и помогает, когда ферритин и воспаление дают противоречивую картину.", ["who_ferritin_2020", "bsg_iron_2021"]),
        ("TSAT", "TSAT показывает доступное циркулирующее железо и помогает отличать железодефицит от железо-ограниченного воспалительного паттерна.", ["who_ferritin_2020", "bsg_iron_2021", "kdigo_anemia_2026"]),
        ("sTfR", "sTfR может быть полезен при подозрении на скрытый дефицит железа, особенно когда ферритин труден для интерпретации.", ["who_ferritin_2020", "bsg_iron_2021"]),
        ("CRP", "CRP нужен, чтобы понимать, может ли ферритин быть повышен как белок острой фазы.", ["who_ferritin_2020", "bsg_iron_2021"]),
        ("ferritin", "Ферритин остаётся ключевым маркером запасов железа, но должен интерпретироваться вместе с воспалением.", ["who_ferritin_2020", "bsg_iron_2021"]),
    ],
    frozenset({"latent_deficiency", "no_anemia_no_deficiency"}): [
        ("ferritin", "Ферритин — наиболее информативный базовый маркер запасов железа при отсутствии выраженного воспаления.", ["who_ferritin_2020", "bsg_iron_2021"]),
        ("TSAT", "TSAT помогает подтвердить ограничение доступного железа при нормальном гемоглобине.", ["who_ferritin_2020", "bsg_iron_2021"]),
        ("Ret_He", "Ret-He отражает железо, доступное развивающимся эритроцитам, и может выявлять раннее ограничение.", ["who_ferritin_2020", "bsg_iron_2021"]),
    ],
    frozenset({"B12_deficiency_anemia", "folate_deficiency_anemia"}): [
        ("MMA", "MMA повышается при B12-дефиците и обычно не повышается при изолированном фолатном дефиците; функцию почек нужно учитывать.", ["nice_b12_2024", "merck_megaloblastic_2026"]),
        ("vitamin_B12", "Сывороточный B12 — базовый маркер направления B12-дефицита.", ["nice_b12_2024"]),
        ("folate", "Фолат помогает отделить фолатный дефицит от B12-дефицита при сходной макроцитарной картине.", ["merck_folate_2026"]),
        ("homocysteine", "Гомоцистеин может повышаться при обоих состояниях и полезен только в сочетании с MMA и уровнями витаминов.", ["merck_megaloblastic_2026"]),
    ],
    frozenset({"B12_deficiency_no_anemia", "folate_deficiency_no_anemia"}): [
        ("MMA", "MMA особенно полезна для различения пограничного B12 и фолатного направления до развития анемии.", ["nice_b12_2024", "merck_megaloblastic_2026"]),
        ("vitamin_B12", "Сывороточный B12 уточняет B12-направление.", ["nice_b12_2024"]),
        ("folate", "Фолат уточняет фолатное направление.", ["merck_folate_2026"]),
        ("homocysteine", "Гомоцистеин неспецифичен и должен интерпретироваться вместе с MMA.", ["merck_megaloblastic_2026"]),
    ],
    frozenset({"anemia_other", "inflammation_anemia"}): [
        ("CRP", "CRP определяет, насколько вероятен воспалительный контекст.", ["who_ferritin_2020", "bsg_iron_2021"]),
        ("TSAT", "TSAT <20% поддерживает железо-ограниченный эритропоэз даже при нормальном/высоком ферритине.", ["kdigo_anemia_2026"]),
        ("ferritin", "Ферритин вместе с CRP и TSAT нужен для интерпретации воспалительного паттерна.", ["who_ferritin_2020", "bsg_iron_2021"]),
        ("eGFR", "рСКФ помогает оценить почечное направление анемии как альтернативу воспалительному.", ["kdigo_anemia_2026"]),
    ],
}


def _generic_reason(feature: str, a: str, b: str) -> str:
    """Типовое обоснование маркера для пар классов без экспертного приоритета."""
    label = LABS.get(feature, {}).get("label", feature)
    return f"{label} заметно различается между группами «{CLASSES.get(a, a)}» и «{CLASSES.get(b, b)}» в предоставленном датасете и может уменьшить модельную неопределённость."


def _ambiguity_factors(patient: dict[str, Any], a: str, b: str) -> list[dict[str, str]]:
    """Клинически объяснимые причины неоднозначности для пары гипотез.

    Например, при паре «железодефицит / воспаление» отмечается, что при CRP > 5 мг/л
    ферритин как белок острой фазы менее специфичен. Статусы: known / ambiguous / missing.
    """
    labs = patient.get("labs", {})
    pair = frozenset({a, b})
    out: list[dict[str, str]] = []
    if pair == frozenset({"iron_deficiency_anemia", "inflammation_anemia"}):
        ferritin = labs.get("ferritin")
        crp = labs.get("CRP")
        tsat = labs.get("TSAT")
        rethe = labs.get("Ret_He")
        stfr = labs.get("sTfR")
        if ferritin is None:
            out.append({"factor": "Ферритин", "state": "missing", "text": "Нет базового маркера запасов железа."})
        elif crp is not None and crp > 5:
            out.append({"factor": "Ферритин", "state": "ambiguous", "text": "Ферритин доступен, но воспаление снижает его специфичность для абсолютного дефицита железа."})
        else:
            out.append({"factor": "Ферритин", "state": "known", "text": "Ферритин доступен и интерпретируется без явного CRP-сигнала воспаления."})
        if crp is None:
            out.append({"factor": "CRP", "state": "missing", "text": "Неизвестно, есть ли воспалительный контекст, влияющий на интерпретацию ферритина."})
        elif crp > 5:
            out.append({"factor": "CRP", "state": "known", "text": f"CRP {crp:g} мг/л поддерживает наличие воспалительного контекста."})
        else:
            out.append({"factor": "CRP", "state": "known", "text": f"CRP {crp:g} мг/л не даёт выраженного воспалительного сигнала по рабочему порогу."})
        if tsat is None:
            out.append({"factor": "TSAT", "state": "missing", "text": "Не хватает оценки доступного циркулирующего железа."})
        if rethe is None:
            out.append({"factor": "Ret-He", "state": "missing", "text": "Не хватает маркера железа, доступного для текущего эритропоэза."})
        if stfr is None:
            out.append({"factor": "sTfR", "state": "missing", "text": "Нет дополнительного маркера, полезного при неоднозначном ферритине."})
    elif pair in {frozenset({"B12_deficiency_anemia", "folate_deficiency_anemia"}), frozenset({"B12_deficiency_no_anemia", "folate_deficiency_no_anemia"})}:
        if labs.get("MMA") is None:
            out.append({"factor": "MMA", "state": "missing", "text": "MMA особенно полезна для отделения B12-направления от изолированного фолатного дефицита; функцию почек нужно учитывать."})
        if labs.get("homocysteine") is None:
            out.append({"factor": "Гомоцистеин", "state": "missing", "text": "Нет общего функционального маркера, который интерпретируется только вместе с MMA и уровнями витаминов."})
        if labs.get("vitamin_B12") is None:
            out.append({"factor": "B12", "state": "missing", "text": "Нет базового маркера B12-направления."})
        if labs.get("folate") is None:
            out.append({"factor": "Фолаты", "state": "missing", "text": "Нет базового маркера фолатного направления."})
    return out


def build_uncertainty_map(patient: dict[str, Any], model_result: dict[str, Any] | None, max_items: int = 5) -> dict[str, Any]:
    """Карта неопределённости по двум ведущим гипотезам модели.

    Для каждого отсутствующего у пациента маркера оценивается, насколько его
    известное значение могло бы разрешить спор между гипотезой №1 и №2.
    Возвращает статус неопределённости, факторы неоднозначности и ранжированный
    список маркеров (``best_next_measurement`` — самый информативный).
    """
    if not model_result or not model_result.get("class_scores"):
        return {
            "status": "unavailable",
            "message": "Карта неопределённости требует обученной модели.",
            "items": [],
        }

    ranked = list(model_result["class_scores"].items())
    if len(ranked) < 2:
        return {"status": "low", "message": "Вторая конкурирующая гипотеза отсутствует.", "items": []}

    (a, pa), (b, pb) = ranked[:2]
    margin = float(pa - pb)
    profile = load_profile()
    class_medians = profile.get("class_medians", {})
    class_cov = profile.get("class_coverage", {})
    missing = [f for f in profile.get("features", {}) if f not in patient["labs"] and f not in {"age_years"}]

    pair = frozenset({a, b})
    priority = PAIR_PRIORITY.get(pair, [])
    priority_index = {feature: (len(priority) - i) for i, (feature, _, _) in enumerate(priority)}
    priority_reason = {feature: reason for feature, reason, _ in priority}
    priority_sources = {feature: sources for feature, _, sources in priority}

    # Stage eligible missing markers first, then score all counterfactual variants
    # in a single model call. Logic is unchanged; only execution is batched.
    candidates: list[dict[str, Any]] = []
    overrides: list[tuple[str, float]] = []
    for feature in missing:
        ma = class_medians.get(a, {}).get(feature)
        mb = class_medians.get(b, {}).get(feature)
        if ma is None or mb is None:
            continue
        ca = float(class_cov.get(a, {}).get(feature, 0.0))
        cb = float(class_cov.get(b, {}).get(feature, 0.0))
        if min(ca, cb) < 0.20:
            continue
        candidates.append({"feature": feature, "ma": float(ma), "mb": float(mb), "ca": ca, "cb": cb})
        overrides.extend([(feature, float(ma)), (feature, float(mb))])

    scored = scores_with_overrides(patient, overrides) if overrides else []
    items: list[dict[str, Any]] = []
    for i, candidate in enumerate(candidates):
        feature = candidate["feature"]
        ma, mb = candidate["ma"], candidate["mb"]
        ca, cb = candidate["ca"], candidate["cb"]
        try:
            sa = scored[2 * i]
            sb = scored[2 * i + 1]
            gap_a = sa.get(a, 0.0) - sa.get(b, 0.0)
            gap_b = sb.get(a, 0.0) - sb.get(b, 0.0)
            counterfactual = abs(gap_a - gap_b)
        except Exception:
            counterfactual = 0.0

        global_info = profile.get("features", {}).get(feature, {})
        p05, p95 = global_info.get("p05"), global_info.get("p95")
        span = (p95 - p05) if p05 is not None and p95 is not None and p95 > p05 else None
        separation = abs(ma - mb) / span if span else 0.0
        coverage = (ca + cb) / 2.0
        # Итоговая информационная ценность маркера: контрфактическое изменение score модели (60%),
        # различие медиан двух классов относительно разброса признака p05–p95 (25%),
        # покрытие признака в датасете (15%) + бонус за экспертный приоритет пары.
        pair_boost = 0.15 * priority_index.get(feature, 0)
        score = (0.60 * counterfactual + 0.25 * separation + 0.15 * coverage) + pair_boost
        if score <= 0.03 and feature not in priority_index:
            continue
        items.append({
            "feature": feature,
            "label": LABS.get(feature, {}).get("label", feature),
            "score": float(score),
            "counterfactual_impact": float(counterfactual),
            "coverage": float(coverage),
            "median_top1": float(ma),
            "median_top2": float(mb),
            "why": priority_reason.get(feature) or _generic_reason(feature, a, b),
            "source_ids": priority_sources.get(feature, ["case_su"]),
        })

    items.sort(key=lambda x: x["score"], reverse=True)
    items = items[:max_items]

    # Статус по отрыву между двумя ведущими классами (рабочие пороги).
    if margin < 0.08:
        status = "high"
        message = "Две гипотезы почти неразличимы по имеющимся данным."
    elif margin < 0.20:
        status = "moderate"
        message = "Есть конкурирующая гипотеза; один-два целевых маркера могут существенно уменьшить неопределённость."
    else:
        status = "low"
        message = "Модельный отрыв заметный, но недостающие данные всё ещё могут изменить дифференциальную картину."

    return {
        "status": status,
        "message": message,
        "top_hypothesis": {"code": a, "label": CLASSES.get(a, a), "score": float(pa)},
        "alternative": {"code": b, "label": CLASSES.get(b, b), "score": float(pb)},
        "margin": margin,
        "ambiguity_factors": _ambiguity_factors(patient, a, b),
        "best_next_measurement": items[0] if items else None,
        "items": items,
        "method_note": (
            "Карта неопределённости состоит из двух частей: клинически объяснимых причин неоднозначности и data-driven ранжирования недостающих признаков. "
            "Приоритет измерений учитывает различие между двумя ведущими классами в предоставленном датасете, покрытие признака и контрфактическое изменение model score. "
            "Это инструмент информационной ценности, а не автоматическое назначение исследования."
        ),
    }
