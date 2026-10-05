"""Справочное сопоставление классов скрининга с кодами МКБ-10."""

from __future__ import annotations

from typing import Any

# ICD-10 suggestions are intentionally family-level. The system never assigns a code
# automatically: coding requires a clinician-confirmed diagnosis and, for dagger/asterisk
# categories, the underlying disease must be coded according to local rules.
ICD10_BY_CLASS: dict[str, list[dict[str, str]]] = {
    "latent_deficiency": [
        {"code": "E61.1", "label": "Дефицит железа", "when": "только после клинического подтверждения дефицита железа без анемии"},
    ],
    "iron_deficiency_anemia": [
        {"code": "D50", "label": "Железодефицитная анемия", "when": "семейство кода; подрубрика зависит от подтверждённой причины"},
        {"code": "D50.0", "label": "Железодефицитная анемия вследствие хронической кровопотери", "when": "только если хроническая кровопотеря подтверждена"},
    ],
    "B12_deficiency_anemia": [
        {"code": "D51", "label": "Витамин-B12-дефицитная анемия", "when": "семейство кода после подтверждения причины"},
    ],
    "B12_deficiency_no_anemia": [
        {"code": "E53.8", "label": "Недостаточность других уточнённых витаминов группы B", "when": "в ICD-10 B12-дефицит без анемии исключён из D51; код требует клинической верификации"},
    ],
    "folate_deficiency_anemia": [
        {"code": "D52", "label": "Фолиеводефицитная анемия", "when": "семейство кода после подтверждения"},
    ],
    "folate_deficiency_no_anemia": [
        {"code": "E53.8", "label": "Недостаточность других уточнённых витаминов группы B", "when": "дефицит фолата без анемии требует клинической верификации"},
    ],
    "B6_deficiency": [
        {"code": "E53.1", "label": "Недостаточность пиридоксина", "when": "после подтверждения дефицита B6"},
    ],
    "copper_deficiency": [
        {"code": "E61.0", "label": "Недостаточность меди", "when": "если подтверждён дефицит меди без установленной нутритивной анемии"},
        {"code": "D53.8", "label": "Другие уточнённые алиментарные анемии", "when": "может применяться при подтверждённой анемии, связанной с дефицитом меди"},
    ],
    "inflammation_anemia": [
        {"code": "D63*", "label": "Анемия при хронических болезнях, классифицированных в других рубриках", "when": "только при подтверждённом основном заболевании; основной код заболевания обязателен"},
    ],
    "mixed_deficiency": [],
    "anemia_other": [
        {"code": "D64.9", "label": "Анемия неуточнённая", "when": "только после клинической оценки и невозможности выбрать более специфичный код"},
    ],
    "no_anemia_no_deficiency": [],
}


def coding_candidates(class_code: str | None) -> dict[str, Any]:
    """Справочные кандидаты МКБ-10 для класса. Это подсказка, а не автоматическое кодирование."""
    if not class_code:
        return {
            "status": "not_applicable",
            "candidates": [],
            "note": "Кодирование МКБ-10 не формируется без рабочей гипотезы.",
            "source_ids": ["who_icd10_2019", "minzdrav_icd10_registry"],
        }
    candidates = ICD10_BY_CLASS.get(class_code, [])
    note = (
        "МКБ-10 здесь используется как справочник для врача, а не как автокодировщик. "
        "Скрининговый класс не равен подтверждённому диагнозу; окончательный код выбирается после клинической верификации."
    )
    if class_code == "mixed_deficiency":
        note += " Для смешанного состояния единого автоматического кода нет: кодируются подтверждённые компоненты и/или установленная анемия."
    return {
        "status": "reference_only",
        "candidates": candidates,
        "note": note,
        "source_ids": ["who_icd10_2019", "minzdrav_icd10_registry"],
    }
