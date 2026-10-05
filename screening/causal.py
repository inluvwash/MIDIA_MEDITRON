"""Причинно-следственное представление результата для врача (гипотезы, не диагнозы)."""

from __future__ import annotations

from typing import Any

from .schema import CLASSES, LABS


def _text(patient: dict[str, Any]) -> str:
    """Единый нижний регистр текста жалоб, анамнеза, препаратов и симптомов для поиска терминов."""
    ctx = patient.get("context", {})
    values = [ctx.get("complaints", ""), ctx.get("history", ""), ctx.get("medications", "")]
    values += [str(x) for x in ctx.get("symptoms", [])]
    return " ".join(values).lower().replace("ё", "е")


def _contains(text: str, *terms: str) -> bool:
    """Есть ли в тексте хотя бы один из корней слов (поиск по подстроке)."""
    return any(term in text for term in terms)


def _marker_nodes(patient: dict[str, Any], code: str) -> list[str]:
    """Ключевые лабораторные узлы цепочки для класса (не более 4, только имеющиеся у пациента)."""
    labs = patient["labs"]
    mapping = {
        "latent_deficiency": ["ferritin", "TSAT", "Ret_He", "sTfR", "RDW"],
        "iron_deficiency_anemia": ["hemoglobin", "ferritin", "TSAT", "Ret_He", "MCV", "RDW"],
        "inflammation_anemia": ["hemoglobin", "CRP", "ferritin", "TSAT", "Ret_He"],
        "B12_deficiency_anemia": ["hemoglobin", "vitamin_B12", "active_B12", "MMA", "homocysteine", "MCV"],
        "B12_deficiency_no_anemia": ["vitamin_B12", "active_B12", "MMA", "homocysteine", "MCV"],
        "folate_deficiency_anemia": ["hemoglobin", "folate", "homocysteine", "MMA", "MCV"],
        "folate_deficiency_no_anemia": ["folate", "homocysteine", "MMA", "MCV"],
        "B6_deficiency": ["vitamin_B6", "hemoglobin", "MCV"],
        "copper_deficiency": ["copper", "ceruloplasmin", "hemoglobin", "WBC"],
        "mixed_deficiency": ["ferritin", "vitamin_B12", "folate", "vitamin_B6", "copper", "RDW", "MCV"],
        "anemia_other": ["hemoglobin", "eGFR", "TSH", "LDH", "indirect_bilirubin", "haptoglobin"],
        "no_anemia_no_deficiency": ["hemoglobin", "ferritin", "vitamin_B12", "folate"],
    }
    nodes = []
    for f in mapping.get(code, []):
        if f in labs:
            unit = patient.get("units", {}).get(f, LABS.get(f, {}).get("unit", ""))
            nodes.append(f"{LABS[f]['label']} = {labs[f]:g} {unit}".strip())
    return nodes[:4]


def _context_causes(patient: dict[str, Any], code: str) -> list[str]:
    # Free-text complaints, anamnesis, medications and selected symptom labels are all used here.
    text = _text(patient)
    causes: list[str] = []
    if code in {"latent_deficiency", "iron_deficiency_anemia", "mixed_deficiency"}:
        if _contains(text, "менорраг", "обильн", "кровопот", "менструац"):
            causes.append("возможная хроническая кровопотеря / обильные менструации")
        if _contains(text, "желуд", "кишеч", "мелена", "черный стул", "чёрный стул", "кровь в стуле"):
            causes.append("возможная желудочно-кишечная кровопотеря")
        if _contains(text, "целиак", "крон", "язвен", "взк", "сибр", "диар"):
            causes.append("возможное нарушение всасывания")
        if _contains(text, "гастрит", "гастрэкт", "резекц"):
            causes.append("желудочно-кишечный контекст, который стоит сопоставить с всасыванием железа")
        if _contains(text, "омепраз", "пантопраз", "эзомепраз", "ипп", "ингибитор протон"):
            causes.append("длительный приём ИПП как возможный контекст для оценки всасывания железа")
    if code in {"B12_deficiency_anemia", "B12_deficiency_no_anemia", "mixed_deficiency"}:
        if _contains(text, "метформин"):
            causes.append("приём метформина как фактор риска B12-дефицита")
        if _contains(text, "омепраз", "пантопраз", "эзомепраз", "ипп"):
            causes.append("длительный приём ИПП как фактор риска B12-дефицита")
        if _contains(text, "веган", "вегетари"):
            causes.append("ограниченное поступление B12 с питанием")
        if _contains(text, "гастрэкт", "резекц", "целиак", "крон", "сибр"):
            causes.append("возможное нарушение всасывания B12")
    if code in {"folate_deficiency_anemia", "folate_deficiency_no_anemia", "mixed_deficiency"}:
        if _contains(text, "алког"):
            causes.append("алкоголь как возможный фактор фолатного дефицита")
        if _contains(text, "метотрекс", "триметоприм"):
            causes.append("лекарственный фактор, влияющий на обмен фолатов")
        if _contains(text, "целиак", "крон", "взк"):
            causes.append("возможное нарушение всасывания фолатов")
    if code == "copper_deficiency" and _contains(text, "бариатр", "резекц", "цинк"):
        causes.append("контекст, совместимый с риском дефицита меди")
    if code == "B6_deficiency" and _contains(text, "изониаз", "алког", "диализ", "малабсорб"):
        causes.append("контекст, совместимый с риском B6-дефицита")
    if code == "inflammation_anemia" and _contains(text, "воспал", "инфекц", "аутоиммун", "онколог", "хронич"):
        causes.append("хроническое воспаление / заболевание как клинический контекст")
    if code == "anemia_other":
        labs = patient["labs"]
        if labs.get("eGFR") is not None and labs["eGFR"] < 60:
            causes.append("почечное направление")
        if labs.get("TSH") is not None and labs["TSH"] > 4.5:
            causes.append("тиреоидное направление")
        if labs.get("LDH", 0) > 250 and labs.get("haptoglobin", 1) < 0.3:
            causes.append("гемолитическое направление")
    return causes[:3]


def build_chains(patient: dict[str, Any], class_codes: list[str]) -> list[dict[str, Any]]:
    """Цепочки «маркеры -> гипотеза -> возможные причины» для переданных классов.

    Причины берутся из контекста (анамнез, препараты) и лабораторных направлений; это
    наводящие связи для врача, а не установленная причинность.
    """
    out = []
    for code in class_codes:
        markers = _marker_nodes(patient, code)
        causes = _context_causes(patient, code)
        if not causes:
            causes = ["причина по доступным данным не определена; требуется сопоставление с анамнезом"]
        hypothesis = CLASSES.get(code, code)
        out.append({
            "class_code": code,
            "label": hypothesis,
            "markers": markers,
            "hypothesis": hypothesis,
            "causes": causes,
            "chain": markers + [hypothesis] + causes,
        })
    return out
