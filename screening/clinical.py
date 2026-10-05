"""Прозрачные клинические правила: порог Hb по ВОЗ, сигналы дефицитов, исключающие причины.

Правила дают класс ``rule_class`` и доказательства (``findings``), которые поддерживают
или оспаривают результат ML-слоя. Пороги указаны в комментариях рядом с правилами;
лабораторные референсы зависят от анализатора и должны уточняться в конкретной лаборатории.
"""

from __future__ import annotations

from typing import Any

from .schema import LABS, anemia_state


def _v(labs: dict[str, float], key: str) -> float | None:
    """Значение показателя или None, если он не измерен."""
    return labs.get(key)


def _finding(code: str, text: str, severity: str = "info", sources: list[str] | None = None) -> dict[str, Any]:
    return {"code": code, "text": text, "severity": severity, "source_ids": sources or []}


def _recommend(code: str, text: str, missing: list[str] | None = None, sources: list[str] | None = None) -> dict[str, Any]:
    return {"code": code, "text": text, "missing_tests": missing or [], "source_ids": sources or []}


def evaluate(patient: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, auditable clinical layer.

    This layer intentionally uses a small set of transparent working thresholds.
    It does not prescribe treatment and it does not pretend to replace the case's
    12-class model. The output is evidence that can support or challenge the ML layer.
    """
    labs = patient["labs"]
    anemia, hb_threshold = anemia_state(patient)
    findings: list[dict[str, Any]] = []
    alerts: list[str] = []
    recommendations: list[dict[str, Any]] = []
    signals: dict[str, dict[str, Any]] = {}

    hb = _v(labs, "hemoglobin")
    mcv = _v(labs, "MCV")
    rdw = _v(labs, "RDW")
    ferritin = _v(labs, "ferritin")
    tsat = _v(labs, "TSAT")
    stfr = _v(labs, "sTfR")
    rethe = _v(labs, "Ret_He")
    crp = _v(labs, "CRP")
    b12 = _v(labs, "vitamin_B12")
    active_b12 = _v(labs, "active_B12")
    mma = _v(labs, "MMA")
    homocysteine = _v(labs, "homocysteine")
    folate = _v(labs, "folate")
    b6 = _v(labs, "vitamin_B6")
    copper = _v(labs, "copper")
    cerulo = _v(labs, "ceruloplasmin")
    egfr = _v(labs, "eGFR")
    tsh = _v(labs, "TSH")
    ldh = _v(labs, "LDH")
    ibili = _v(labs, "indirect_bilirubin")
    hapto = _v(labs, "haptoglobin")

    findings.append(_finding(
        "hb_threshold",
        f"Гемоглобин {hb:g} г/л; порог анемии по ТЗ для указанного пола — менее {hb_threshold:g} г/л. "
        + ("Признак анемии есть." if anemia else "По этому порогу анемии нет."),
        "warning" if anemia else "info",
        ["case_su", "who_hb_2024"],
    ))

    # Iron signal. WHO 2020 explicitly warns that inflammation changes ferritin interpretation.
    # The case also asks to jointly consider ferritin, TSAT, sTfR and Ret-He rather than a single marker.
    iron_evidence: list[str] = []
    iron_strength = 0.0
    inflammation_present = crp is not None and crp > 5
    if ferritin is not None:
        if inflammation_present and ferritin < 70:
            iron_strength += 3.0
            iron_evidence.append(
                f"ферритин {ferritin:g} мкг/л < 70 при наличии воспалительного сигнала; WHO допускает такой порог у взрослых с инфекцией/воспалением"
            )
        elif not inflammation_present and ferritin < 15:
            iron_strength += 3.0
            iron_evidence.append(f"ферритин {ferritin:g} мкг/л выраженно снижен")
        elif not inflammation_present and ferritin < 30:
            iron_strength += 2.5
            iron_evidence.append(f"ферритин {ferritin:g} мкг/л снижен по рабочему клиническому порогу 30")
        elif ferritin < 45:
            iron_strength += 1.0
            iron_evidence.append(f"ферритин {ferritin:g} мкг/л находится в пограничной зоне")
    if tsat is not None and tsat < 20:
        iron_strength += 2.0
        iron_evidence.append(f"TSAT {tsat:g}% < 20%")
    if rethe is not None and rethe < 28:
        iron_strength += 1.0
        iron_evidence.append(f"Ret-He {rethe:g} пг снижен по рабочему порогу 28 пг; референс зависит от анализатора")
    if stfr is not None and stfr > 4.4:
        iron_strength += 1.0
        iron_evidence.append(f"sTfR {stfr:g} мг/л повышен по рабочему assay-зависимому порогу")
    signals["iron"] = {"strength": iron_strength, "positive": iron_strength >= 3, "evidence": iron_evidence}
    if iron_evidence:
        findings.append(_finding(
            "iron_pattern",
            "Железо: " + "; ".join(iron_evidence) + ".",
            "warning" if iron_strength >= 3 else "info",
            ["who_ferritin_2020", "bsg_iron_2021", "case_su"],
        ))

    # B12. MMA is used as confirmation in low-normal serum B12, but renal dysfunction can elevate it.
    b12_evidence: list[str] = []
    b12_strength = 0
    # NICE NG239 (2024): total B12 <180 ng/L supports deficiency; 180-350 is
    # indeterminate. Active B12 <25 pmol/L supports deficiency; 25-70 is
    # indeterminate. Local validated laboratory thresholds should override these
    # generic cut-offs when available.
    if b12 is not None and b12 < 180:
        b12_strength += 3
        b12_evidence.append(f"B12 {b12:g} пг/мл < 180 (диапазон, поддерживающий дефицит по NICE NG239)")
    elif b12 is not None and b12 <= 350:
        b12_strength += 1
        b12_evidence.append(f"B12 {b12:g} пг/мл в неопределённой зоне 180–350 по NICE NG239")
    if active_b12 is not None and active_b12 < 25:
        b12_strength += 2
        b12_evidence.append(f"активный B12 {active_b12:g} пмоль/л < 25")
    elif active_b12 is not None and active_b12 <= 70:
        b12_strength += 1
        b12_evidence.append(f"активный B12 {active_b12:g} пмоль/л в неопределённой зоне 25–70")
    if mma is not None and mma > 0.4:
        if egfr is not None and egfr < 60:
            b12_evidence.append(f"MMA {mma:g} мкмоль/л повышена, но рСКФ {egfr:g} может снижать специфичность")
            b12_strength += 1
        else:
            b12_strength += 2
            b12_evidence.append(f"MMA {mma:g} мкмоль/л выше рабочего порога 0.4; итоговую интерпретацию следует сверять с референсом лаборатории")
    if homocysteine is not None and homocysteine > 15:
        b12_strength += 1
        b12_evidence.append(f"гомоцистеин {homocysteine:g} мкмоль/л повышен и неспецифичен")
    signals["b12"] = {"strength": b12_strength, "positive": b12_strength >= 3, "evidence": b12_evidence}
    if b12_evidence:
        findings.append(_finding("b12_pattern", "B12: " + "; ".join(b12_evidence) + ".", "warning" if b12_strength >= 3 else "info", ["nice_b12_2024", "merck_megaloblastic_2026"]))

    # Folate.
    folate_evidence: list[str] = []
    folate_strength = 0
    if folate is not None and folate < 3:
        folate_strength += 3
        folate_evidence.append(f"фолаты {folate:g} нг/мл < 3")
    elif folate is not None and folate < 4:
        folate_strength += 1
        folate_evidence.append(f"фолаты {folate:g} нг/мл погранично снижены")
    if homocysteine is not None and homocysteine > 15:
        folate_strength += 1
        folate_evidence.append("гомоцистеин повышен, что совместимо как с фолатным, так и с B12-дефицитом")
    if mma is not None and mma <= 0.4 and homocysteine is not None and homocysteine > 15:
        folate_strength += 1
        folate_evidence.append("MMA не повышена при высоком гомоцистеине — это поддерживает фолатное направление сильнее B12")
    signals["folate"] = {"strength": folate_strength, "positive": folate_strength >= 3, "evidence": folate_evidence}
    if folate_evidence:
        findings.append(_finding("folate_pattern", "Фолаты: " + "; ".join(folate_evidence) + ".", "warning" if folate_strength >= 3 else "info", ["merck_folate_2026", "merck_megaloblastic_2026"]))

    # B6 and copper are part of the official dataset classes. We keep the rule layer conservative.
    b6_evidence: list[str] = []
    b6_strength = 0
    if b6 is not None and b6 < 20:
        b6_strength = 3
        b6_evidence.append(f"PLP/B6 {b6:g} нмоль/л ниже рабочего порога 20")
    signals["b6"] = {"strength": b6_strength, "positive": b6_strength >= 3, "evidence": b6_evidence}
    if b6_evidence:
        findings.append(_finding("b6_pattern", "B6: " + "; ".join(b6_evidence) + ".", "warning", ["merck_b6_2026", "case_su"]))

    copper_evidence: list[str] = []
    copper_strength = 0
    if copper is not None and copper < 11:
        copper_strength += 3
        copper_evidence.append(f"медь {copper:g} мкмоль/л ниже рабочего порога 11")
    if cerulo is not None and cerulo < 0.20:
        copper_strength += 1
        copper_evidence.append(f"церулоплазмин {cerulo:g} г/л снижен")
    signals["copper"] = {"strength": copper_strength, "positive": copper_strength >= 3, "evidence": copper_evidence}
    if copper_evidence:
        findings.append(_finding("copper_pattern", "Медь: " + "; ".join(copper_evidence) + ".", "warning", ["merck_copper_2025", "case_su"]))

    nutrient_positive = [k for k in ("iron", "b12", "folate", "b6", "copper") if signals[k]["positive"]]

    inflammation_evidence: list[str] = []
    inflammation_strength = 0
    if crp is not None and crp > 5:
        inflammation_strength += 1
        inflammation_evidence.append(f"CRP {crp:g} мг/л > 5")
    if ferritin is not None and ferritin >= 100:
        inflammation_strength += 1
        inflammation_evidence.append(f"ферритин {ferritin:g} мкг/л не снижен")
    if tsat is not None and tsat < 20:
        inflammation_strength += 1
        inflammation_evidence.append(f"TSAT {tsat:g}% < 20%")
    inflammation_positive = anemia and inflammation_strength >= 3 and not nutrient_positive
    signals["inflammation"] = {"strength": inflammation_strength, "positive": inflammation_positive, "evidence": inflammation_evidence}
    if inflammation_evidence:
        findings.append(_finding(
            "inflammation_pattern",
            "Воспалительный / железо-ограниченный паттерн: " + "; ".join(inflammation_evidence) + ". "
            "Ферритин является белком острой фазы, поэтому его нормальное или повышенное значение не исключает ограничение доступности железа.",
            "warning" if inflammation_positive else "info",
            ["who_ferritin_2020", "bsg_iron_2021", "kdigo_anemia_2026"],
        ))

    # Morphology and mixed deficiency hints.
    if rdw is not None and rdw > 15 and mcv is not None and 80 <= mcv <= 100:
        findings.append(_finding(
            "normal_mcv_high_rdw",
            f"MCV {mcv:g} фл нормоцитарный, но RDW {rdw:g}% повышен: разнонаправленные дефициты могут маскировать изменение среднего объёма эритроцита.",
            "info",
            ["case_su", "merck_megaloblastic_2026"],
        ))

    # Alternative causes / exclusion layer.
    exclusions: list[dict[str, Any]] = []
    if egfr is None:
        exclusions.append({"title": "Почечная функция", "status": "incomplete", "missing": ["eGFR", "creatinine"]})
    elif egfr < 60:
        exclusions.append({"title": "Почечная функция", "status": "abnormal", "missing": [], "text": f"рСКФ {egfr:g} мл/мин/1.73м²"})
        findings.append(_finding("renal", f"рСКФ {egfr:g}: почечная дисфункция может участвовать в анемическом паттерне и влиять на интерпретацию MMA.", "warning", ["kdigo_anemia_2026"]))
    else:
        exclusions.append({"title": "Почечная функция", "status": "no_screening_signal", "missing": []})

    hemolysis_available = sum(v is not None for v in (ldh, ibili, hapto))
    hemolysis_abnormal = ((ldh is not None and ldh > 250) + (ibili is not None and ibili > 17) + (hapto is not None and hapto < 0.3)) >= 2
    if hemolysis_available < 2:
        missing = [k for k, v in (("LDH", ldh), ("indirect_bilirubin", ibili), ("haptoglobin", hapto)) if v is None]
        exclusions.append({"title": "Гемолиз", "status": "incomplete", "missing": missing})
    elif hemolysis_abnormal:
        exclusions.append({"title": "Гемолиз", "status": "abnormal", "missing": []})
        findings.append(_finding("hemolysis", "Есть сочетание лабораторных признаков, совместимое с гемолизом; это альтернативное направление причин анемии.", "warning", ["case_su"]))
    else:
        exclusions.append({"title": "Гемолиз", "status": "no_screening_signal", "missing": []})

    if tsh is None:
        exclusions.append({"title": "Щитовидная железа", "status": "incomplete", "missing": ["TSH"]})
    elif tsh > 4.5:
        exclusions.append({"title": "Щитовидная железа", "status": "abnormal", "missing": [], "text": f"TSH {tsh:g}"})
        findings.append(_finding("thyroid", f"ТТГ {tsh:g} мМЕ/л повышен по рабочему порогу; гипотиреоидное направление требует клинической проверки.", "info", ["case_su"]))
    else:
        exclusions.append({"title": "Щитовидная железа", "status": "no_screening_signal", "missing": []})

    # Red flags are deterministic and independent from the ML/LLM layer.
    if hb is not None and hb < 90:
        alerts.append(f"Выраженное снижение гемоглобина: {hb:g} г/л. Требуется очная клиническая оценка срочности.")
    if mcv is not None and mcv < 70:
        alerts.append(f"Выраженный микроцитоз: MCV {mcv:g} фл. Нужна очная оценка причины.")
    if mcv is not None and mcv > 110:
        alerts.append(f"Выраженный макроцитоз: MCV {mcv:g} фл. Нужна очная оценка причины.")
    if hemolysis_abnormal:
        alerts.append("Лабораторный паттерн может соответствовать гемолизу; требуется врачебная оценка и подтверждение.")

    # Rule-only 12-class fallback. The ML layer remains a separate, visible layer.
    if len(nutrient_positive) >= 2:
        rule_class = "mixed_deficiency"
    elif signals["iron"]["positive"]:
        rule_class = "iron_deficiency_anemia" if anemia else "latent_deficiency"
    elif signals["b12"]["positive"]:
        rule_class = "B12_deficiency_anemia" if anemia else "B12_deficiency_no_anemia"
    elif signals["folate"]["positive"]:
        rule_class = "folate_deficiency_anemia" if anemia else "folate_deficiency_no_anemia"
    elif signals["b6"]["positive"]:
        rule_class = "B6_deficiency"
    elif signals["copper"]["positive"]:
        rule_class = "copper_deficiency"
    elif inflammation_positive:
        rule_class = "inflammation_anemia"
    elif anemia:
        rule_class = "anemia_other"
    else:
        complete_core = all(k in labs for k in ("ferritin", "TSAT", "vitamin_B12", "folate"))
        rule_class = "no_anemia_no_deficiency" if complete_core else None

    # Recommendations: formulate as information gain, not a shopping list of tests.
    if iron_strength in {1, 2} or (anemia and crp is not None and crp > 5 and (tsat is None or rethe is None)):
        missing = [k for k in ("TSAT", "Ret_He", "sTfR") if k not in labs]
        if missing:
            recommendations.append(_recommend(
                "iron_vs_inflammation",
                "Для различения абсолютного дефицита железа и воспалительного ограничения железа полезнее всего заполнить недостающие маркеры доступности железа, а не повторять только ферритин.",
                missing,
                ["who_ferritin_2020", "bsg_iron_2021", "kdigo_anemia_2026"],
            ))
    if b12 is not None and 180 <= b12 <= 350 and mma is None:
        recommendations.append(_recommend(
            "b12_confirmation",
            "B12 находится в пограничной зоне; MMA сильнее уменьшит неопределённость между B12-дефицитом и альтернативами. Интерпретация MMA зависит от функции почек.",
            ["MMA"],
            ["nice_b12_2024", "merck_megaloblastic_2026"],
        ))
    if anemia and rule_class == "anemia_other":
        missing = [k for k in ("eGFR", "TSH", "LDH", "indirect_bilirubin", "haptoglobin") if k not in labs]
        if missing:
            recommendations.append(_recommend(
                "other_anemia_workup",
                "После исключения основных дефицитов следующий шаг — проверить альтернативные направления из ТЗ: почечную функцию, гемолиз и функцию щитовидной железы.",
                missing,
                ["case_su", "kdigo_anemia_2026"],
            ))

    return {
        "anemia": anemia,
        "threshold": hb_threshold,
        "hemoglobin": hb,
        "signals": signals,
        "findings": findings,
        "alerts": alerts,
        "exclusions": exclusions,
        "recommendations": recommendations,
        "rule_class": rule_class,
        "candidate_deficits": nutrient_positive,
    }
