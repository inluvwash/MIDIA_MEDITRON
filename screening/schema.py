"""Контракт данных: справочник показателей, единицы измерения, нормализация входа.

Справочник загружается из ``knowledge/`` (словарь стандартизации и описание переменных).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "knowledge" / "standardization_dict.json"
VARIABLES_PATH = ROOT / "knowledge" / "variables.json"

with open(SCHEMA_PATH, encoding="utf-8") as f:
    _SCHEMA = json.load(f)
with open(VARIABLES_PATH, encoding="utf-8") as f:
    _VARIABLES_DOC = json.load(f)

VARIABLES: list[dict[str, Any]] = list(_VARIABLES_DOC.get("variables", []))
VARIABLE_BY_NAME: dict[str, dict[str, Any]] = {v["variable"]: v for v in VARIABLES}
LABS: dict[str, dict[str, Any]] = _SCHEMA["fields"]
# variables.xlsx is the organiser-provided data dictionary.  We enrich the runtime
# schema with its descriptions/original units while keeping canonical Russian units
# in standardization_dict.json for normalization and UI.
for _name, _meta in LABS.items():
    _source = VARIABLE_BY_NAME.get(_name, {})
    if _source:
        _meta.setdefault("description", _source.get("description", ""))
        _meta.setdefault("source_unit", _source.get("source_unit", ""))
FEATURES: list[str] = list(_SCHEMA["model_features"])
EXCLUDED_TARGETS: set[str] = set(_SCHEMA["excluded_targets"])
ALIASES: dict[str, str] = dict(_SCHEMA["aliases"])
ANEMIA_THRESHOLDS = dict(_SCHEMA.get("anemia_thresholds", {"F": 120, "M": 130}))

CLASSES = {
    "no_anemia_no_deficiency": "Нет анемии и убедительных признаков исследуемых дефицитов",
    "latent_deficiency": "Латентное дефицитное состояние без анемии",
    "iron_deficiency_anemia": "Железодефицитная анемия",
    "B12_deficiency_anemia": "B12-дефицитная анемия",
    "B12_deficiency_no_anemia": "Дефицит B12 без анемии",
    "folate_deficiency_anemia": "Фолат-дефицитная анемия",
    "folate_deficiency_no_anemia": "Дефицит фолата без анемии",
    "B6_deficiency": "Дефицит витамина B6",
    "copper_deficiency": "Дефицит меди",
    "inflammation_anemia": "Анемия с воспалительным / железо-ограниченным паттерном",
    "mixed_deficiency": "Сочетанное дефицитное состояние",
    "anemia_other": "Анемия иной или пока не уточнённой природы",
}

CLASS_ORDER = list(CLASSES)

CLASS_REQUIRES_ANEMIA = {
    "iron_deficiency_anemia",
    "B12_deficiency_anemia",
    "folate_deficiency_anemia",
    "inflammation_anemia",
    "anemia_other",
}
CLASS_REQUIRES_NO_ANEMIA = {
    "no_anemia_no_deficiency",
    "latent_deficiency",
    "B12_deficiency_no_anemia",
    "folate_deficiency_no_anemia",
}

CONTEXT_FIELDS = {"complaints", "symptoms", "history", "medications"}

SYMPTOM_LABELS = {
    "fatigue": "Утомляемость / слабость",
    "reduced_tolerance": "Снижение переносимости физической нагрузки",
    "dyspnea": "Одышка при нагрузке",
    "dizziness": "Головокружение",
    "presyncope": "Предобморочное состояние / обморок",
    "palpitations": "Сердцебиение",
    "headache": "Головная боль",
    "pallor": "Бледность кожи / слизистых",
    "cold_intolerance": "Зябкость / непереносимость холода",
    "pica": "Извращение вкуса / пика",
    "hair_loss": "Выпадение волос / ломкость ногтей",
    "restless_legs": "Синдром беспокойных ног",
    "paresthesia": "Парестезии / покалывание",
    "numbness": "Онемение / снижение чувствительности",
    "gait": "Нарушение походки / равновесия",
    "memory": "Снижение концентрации / памяти",
    "glossitis": "Глоссит / жжение языка",
    "stomatitis": "Стоматит / изменения слизистой полости рта",
    "appetite": "Снижение аппетита",
    "weight_loss": "Непреднамеренная потеря веса",
    "diarrhea": "Хроническая диарея",
    "abdominal_pain": "Боль / дискомфорт в животе",
    "melena": "Чёрный стул / мелена",
    "blood_stool": "Кровь в стуле",
    "heavy_menses": "Обильные / длительные менструации",
    "other_bleeding": "Другие признаки кровопотери",
    "fever": "Длительная лихорадка / субфебрилитет",
    "joint_pain": "Суставные боли / признаки хронического воспаления",
}

DEFAULT_LABS = [
    "hemoglobin", "MCV", "RDW", "ferritin", "TSAT", "CRP", "vitamin_B12", "folate",
]

DISCLAIMER = (
    "Прототип системы поддержки врачебных решений для врачей. Область применения текущей версии: "
    "взрослые пациенты вне беременности; сервис не предназначен для неотложных состояний. "
    "Результат является скрининговой гипотезой, а не подтверждённым диагнозом; лечение и дозировки сервис не назначает."
)

UNIT_NOTE = (
    "Единицы нормализуются только для известных преобразований. Если единица неизвестна, "
    "показатель не используется в расчёте до ручной проверки."
)


class InputError(ValueError):
    pass


def _key(text: Any) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9%]+", "", text)


def field(name: Any) -> str | None:
    if name in LABS or name in {"age_years", "sex", "patient_id"}:
        return str(name)
    return ALIASES.get(_key(name))


SEX_MAP = {
    "f": "F", "female": "F", "жен": "F", "ж": "F", "женский": "F",
    "m": "M", "male": "M", "муж": "M", "м": "M", "мужской": "M",
}


def normalize_sex(value: Any) -> str:
    s = str(value or "").strip().lower().replace("ё", "е")
    out = SEX_MAP.get(s)
    if out not in {"F", "M"}:
        raise InputError("Укажите пол")
    return out


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    s = str(value).strip().replace("\u00a0", " ")
    if not s or s.lower() in {"nan", "none", "null", "-", "—"}:
        return None
    s = s.replace(" ", "").replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _norm_unit(unit: Any) -> str:
    s = str(unit or "").strip().lower().replace("µ", "мк").replace("μ", "мк")
    # Пробелы несущественны, но степень (^) сохраняем: 10^12/л != 1012/л.
    s = s.replace(" ", "")
    repl = {
        "g/l": "г/л", "гл": "г/л", "mg/l": "мг/л", "мгл": "мг/л",
        "ug/l": "мкг/л", "mcg/l": "мкг/л", "мкгл": "мкг/л",
        "ng/ml": "нг/мл", "нгмл": "нг/мл", "pg/ml": "пг/мл", "пгмл": "пг/мл",
        "pmol/l": "пмоль/л", "пмольл": "пмоль/л", "umol/l": "мкмоль/л",
        "мкмольл": "мкмоль/л", "mmol/l": "ммоль/л", "ммольл": "ммоль/л",
        "fl": "фл", "fL": "фл", "%": "%", "u/l": "ед/л", "едл": "ед/л",
        "me/l": "ме/л", "мме/л": "мме/л", "мме/l": "мме/л", "мед/л": "мме/л",
        "ml/min/1.73m2": "мл/мин/1.73м2", "мл/мин/1.73м2": "мл/мин/1.73м2",
        "10^12/l": "10^12/л", "10^9/l": "10^9/л",
        "1012/l": "10^12/л", "109/l": "10^9/л",
        # Частые обозначения из российских лабораторных бланков.
        "млн/мкл": "10^12/л", "мнл/мкл": "10^12/л", "млн/мкл.": "10^12/л",
        "10*6/мкл": "10^12/л", "106/мкл": "10^12/л",
        "тыс/мкл": "10^9/л", "тыс/мкл.": "10^9/л",
        "10*3/мкл": "10^9/л", "103/мкл": "10^9/л",
    }
    return repl.get(s, s)


# Conservative, explicit conversions only. Canonical units come from standardization_dict.json.
_CONVERSIONS: dict[tuple[str, str], float] = {
    ("г/дл", "г/л"): 10.0,
    ("mg/dl", "мг/л"): 10.0,
    ("мг/дл", "мг/л"): 10.0,
    ("нг/мл", "мкг/л"): 1.0,
    ("мкг/л", "нг/мл"): 1.0,
}


def convert_value(name: str, value: Any, unit: Any, units_confirmed: bool = False) -> tuple[float | None, str, str | None]:
    """Return normalized value, canonical unit, optional warning."""
    if name not in LABS:
        raise InputError(f"Неизвестный лабораторный показатель: {name}")
    num = _num(value)
    canonical = _norm_unit(LABS[name].get("unit", ""))
    if num is None:
        return None, canonical, None
    raw_unit = _norm_unit(unit)
    if not raw_unit:
        if not units_confirmed:
            return None, canonical, f"{LABS[name]['label']}: не указана единица"
        raw_unit = canonical
    if raw_unit == canonical:
        return num, canonical, None

    # Selected clinically common conversions.
    if name == "hemoglobin" and raw_unit in {"g/dl", "г/дл"} and canonical == "г/л":
        return num * 10.0, canonical, None
    if name in {"ferritin"} and raw_unit == "нг/мл" and canonical == "мкг/л":
        return num, canonical, None
    if name == "vitamin_B12" and raw_unit == "пмоль/л" and canonical == "пг/мл":
        return num * 1.355, canonical, "B12: преобразовано из пмоль/л в пг/мл коэффициентом 1.355"
    if name == "folate" and raw_unit == "нмоль/л" and canonical == "нг/мл":
        return num / 2.266, canonical, "Фолаты: преобразовано из нмоль/л в нг/мл"
    if name == "creatinine" and raw_unit in {"мг/дл", "mg/dl"} and canonical == "мкмоль/л":
        return num * 88.4, canonical, "Креатинин: преобразовано из мг/дл в мкмоль/л"

    factor = _CONVERSIONS.get((raw_unit, canonical))
    if factor is not None:
        return num * factor, canonical, None
    return None, canonical, f"{LABS[name]['label']}: неподдерживаемая единица '{unit}' (ожидается {canonical})"


def normalize_record(raw: dict[str, Any], units_confirmed: bool = False) -> dict[str, Any]:
    """Привести сырую карточку к контракту сервиса.

    Проверяет возраст (взрослые 18–120 лет) и пол, сопоставляет названия показателей
    с каноническими, переводит значения в канонические единицы и отбрасывает целевые
    колонки датасета (они не должны попадать в признаки). Показатели без указанной
    единицы пропускаются, пока врач не подтвердит единицы (``units_confirmed``).
    """
    if not isinstance(raw, dict):
        raise InputError("Карточка пациента должна быть объектом")

    merged: dict[str, Any] = dict(raw)
    if isinstance(raw.get("labs"), dict):
        merged.update(raw["labs"])

    age_raw = merged.get("age_years", merged.get("Возраст", merged.get("age")))
    age = _num(age_raw)
    if age is None:
        raise InputError("Укажите возраст")
    if age < 18:
        raise InputError("Проект предназначен только для взрослых: возраст должен быть не менее 18 лет")
    if age > 120:
        raise InputError("Возраст вне допустимого технического диапазона 18–120 лет")

    sex = normalize_sex(merged.get("sex", merged.get("Пол", merged.get("gender"))))

    pregnant_raw = raw.get("pregnant", raw.get("Беременность", False))
    if isinstance(pregnant_raw, str):
        pregnant = pregnant_raw.strip().lower().replace("ё", "е") in {"1", "true", "yes", "y", "да", "беременна"}
    else:
        pregnant = bool(pregnant_raw)
    if pregnant:
        raise InputError(
            "Текущая версия не применяется при беременности: для неё нужны отдельные физиологические пороги и клинический профиль."
        )

    patient_id = str(merged.get("patient_id", merged.get("Код пациента", "anonymous")) or "anonymous").strip()

    units = raw.get("units", {}) if isinstance(raw.get("units"), dict) else {}
    labs: dict[str, float] = {}
    lab_units: dict[str, str] = {}
    warnings: list[str] = []
    seen: set[str] = set()

    for k, v in merged.items():
        canonical = field(k)
        if canonical not in LABS:
            continue
        if canonical in seen:
            raise InputError(f"Повтор лабораторного показателя: {LABS[canonical]['label']}")
        seen.add(canonical)
        if isinstance(v, dict):
            value = v.get("value")
            unit = v.get("unit", units.get(k, ""))
        else:
            value = v
            unit = units.get(k, LABS[canonical].get("unit", "") if units_confirmed else "")
        val, canonical_unit, warning = convert_value(canonical, value, unit, units_confirmed=units_confirmed)
        if warning:
            warnings.append(warning)
        if val is None:
            continue
        max_value = LABS[canonical].get("max")
        if val < 0 or (max_value is not None and val > float(max_value)):
            raise InputError(f"{LABS[canonical]['label']}: значение {val:g} вне технического диапазона")
        labs[canonical] = float(val)
        lab_units[canonical] = canonical_unit

    if "hemoglobin" not in labs:
        raise InputError("Для скрининга обязателен гемоглобин")

    symptoms = raw.get("symptoms", [])
    if isinstance(symptoms, str):
        symptoms = [s.strip() for s in re.split(r"[,;\n]+", symptoms) if s.strip()]
    elif not isinstance(symptoms, list):
        symptoms = []

    return {
        "patient_id": patient_id,
        "age_years": float(age),
        "sex": sex,
        "labs": labs,
        "units": lab_units,
        "warnings": warnings,
        "context": {
            "complaints": str(raw.get("complaints", raw.get("Жалобы", "")) or "").strip(),
            "symptoms": [str(x) for x in symptoms],
            "history": str(raw.get("history", raw.get("Хронические заболевания", "")) or "").strip(),
            "medications": str(raw.get("medications", raw.get("Постоянные препараты", "")) or "").strip(),
        },
    }


def anemia_state(patient: dict[str, Any]) -> tuple[bool, float]:
    """Критерий анемии ВОЗ: гемоглобин ниже порога для пола (Ж < 120, М < 130 г/л).

    Возвращает признак анемии и использованный порог.
    """
    threshold = float(ANEMIA_THRESHOLDS[patient["sex"]])
    return patient["labs"]["hemoglobin"] < threshold, threshold


def feature_row(patient: dict[str, Any]) -> dict[str, float]:
    """Вектор признаков модели; пол кодируется M=1, F=0, пропуски остаются NaN."""
    labs = patient["labs"]
    row: dict[str, float] = {}
    for name in FEATURES:
        if name == "age_years":
            row[name] = float(patient["age_years"])
        elif name == "sex":
            row[name] = 1.0 if patient["sex"] == "M" else 0.0
        else:
            row[name] = labs.get(name, float("nan"))
    return row


def class_is_consistent(code: str, anemia: bool) -> bool:
    """Совместим ли класс с наличием/отсутствием анемии по критерию ВОЗ."""
    if code in CLASS_REQUIRES_ANEMIA:
        return anemia
    if code in CLASS_REQUIRES_NO_ANEMIA:
        return not anemia
    return True
