"""Локальный импорт лабораторных файлов без Ollama и внешних сервисов.

Поддерживаются JSON, CSV/TSV/TXT, XLSX/XLS, HTML и текстовые PDF.
Формат определяется прежде всего по содержимому, а не по расширению.
Таблицы могут быть:
- широкими: одна строка пациента, показатели в отдельных колонках;
- длинными: Показатель | Результат | Единица;
- с несколькими служебными строками перед заголовком.
"""

from __future__ import annotations

import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pypdf import PdfReader

from .schema import LABS, field

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 1000

_NAME_HEADERS = {
    "показатель", "исследование", "анализ", "тест", "наименование",
    "наименованиепоказателя", "параметр", "indicator", "test", "analyte", "name",
}
_VALUE_HEADERS = {
    "значение", "результат", "результаты", "result", "value", "measurement",
}
_UNIT_HEADERS = {
    "единица", "единицы", "едизм", "единицаизмерения", "единицыизмерения",
    "unit", "units", "measurementunit",
}
_PATIENT_HEADERS = {"patientid", "кодпациента", "идпациента", "инз", "id"}

# Метки для PDF и свободного текста. Более длинные варианты проверяются первыми.
_PDF_ALIASES: dict[str, list[str]] = {
    "hemoglobin": ["гемоглобин", "hemoglobin", "hgb", "hb"],
    "RBC": ["эритроциты", "red blood cells", "rbc"],
    "hematocrit": ["гематокрит", "hematocrit", "hct"],
    "MCV": ["mcv"],
    "MCHC": ["mchc"],
    "MCH": ["mch"],
    "RDW": ["rdw-cv", "rdw"],
    "platelets": ["тромбоциты", "platelets", "plt"],
    "WBC": ["лейкоциты", "white blood cells", "wbc"],
    "reticulocytes": ["ретикулоциты", "reticulocytes"],
    "ferritin": ["ферритин", "ferritin"],
    "serum_iron": ["сывороточное железо", "железо", "serum iron"],
    "transferrin": ["трансферрин", "transferrin"],
    "TIBC": ["ожсс", "tibc"],
    "UIBC": ["лжсс", "uibc"],
    "TSAT": ["насыщение трансферрина", "кнт", "tsat"],
    "sTfR": ["растворимый рецептор трансферрина", "stfr"],
    "Ret_He": ["ret-he", "ret he", "chr"],
    "vitamin_B12": ["витамин b12", "vitamin b12", "b12"],
    "active_B12": ["активный b12", "голотранскобаламин", "holotc"],
    "MMA": ["метилмалоновая кислота", "mma"],
    "homocysteine": ["гомоцистеин", "homocysteine"],
    "folate": ["фолиевая кислота", "фолат", "folate"],
    "vitamin_B6": ["витамин b6", "vitamin b6", "plp", "p5p"],
    "copper": ["медь", "copper"],
    "ceruloplasmin": ["церулоплазмин", "ceruloplasmin"],
    "CRP": ["с-реактивный белок", "срб", "crp"],
    "ESR": ["соэ (по вестергрену)", "соэ", "esr"],
    "creatinine": ["креатинин", "creatinine"],
    "eGFR": ["рскф", "скф", "egfr"],
    "TSH": ["ттг", "tsh"],
    "albumin": ["альбумин", "albumin"],
    "LDH": ["лдг", "ldh"],
    "indirect_bilirubin": ["непрямой билирубин", "indirect bilirubin"],
    "haptoglobin": ["гаптоглобин", "haptoglobin"],
}

_UNIT_RE = re.compile(
    r"(?:10\s*[\^*]?\s*(?:9|12)\s*/\s*[лl]|"
    r"(?:тыс|млн|мнл)\s*/\s*(?:мкл|мкл)|"
    r"г\s*/\s*(?:дл|л)|мг\s*/\s*л|мкг\s*/\s*л|нг\s*/\s*мл|пг\s*/\s*мл|"
    r"мкмоль\s*/\s*л|ммоль\s*/\s*л|пмоль\s*/\s*л|нмоль\s*/\s*л|"
    r"фл|пг|%|мм\s*/\s*ч|ед\s*/\s*л|г\s*/\s*л|мл\s*/\s*мин\s*/\s*1[.,]73\s*м2)",
    flags=re.I,
)


def _check_size(data: bytes) -> None:
    if len(data) > MAX_BYTES:
        raise ValueError("Файл превышает лимит 20 МБ")
    if not data:
        raise ValueError("Файл пуст")


def _key(value: Any) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    return re.sub(r"[^a-zа-я0-9]+", "", text)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null", "—", "-"}


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(how="all").dropna(axis=1, how="all")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Файл содержит более {MAX_ROWS} строк")
    return frame


def _find_header_row(raw: pd.DataFrame) -> int | None:
    """Ищет строку заголовка, если перед таблицей есть шапка лаборатории."""
    for idx in range(min(30, len(raw))):
        keys = {_key(v) for v in raw.iloc[idx].tolist() if not _is_empty(v)}
        has_name = bool(keys & _NAME_HEADERS)
        has_value = bool(keys & _VALUE_HEADERS)
        # Также принимаем широкую таблицу, если в строке есть минимум 2 канонических поля.
        canonical_count = sum(1 for v in raw.iloc[idx].tolist() if field(v) is not None)
        if (has_name and has_value) or canonical_count >= 2:
            return idx
    return None


def _prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    raw = _clean_frame(raw)
    header_idx = _find_header_row(raw)
    if header_idx is None:
        # Если pandas уже прочитал нормальные заголовки — оставляем как есть.
        if sum(1 for c in raw.columns if field(c) is not None or _key(c) in _NAME_HEADERS | _VALUE_HEADERS) >= 1:
            return raw
        return raw
    headers = [str(v).strip() if not _is_empty(v) else f"column_{i}" for i, v in enumerate(raw.iloc[header_idx])]
    frame = raw.iloc[header_idx + 1 :].copy()
    frame.columns = headers
    return _clean_frame(frame)


def _column_by_header(frame: pd.DataFrame, accepted: set[str]) -> str | None:
    return next((str(c) for c in frame.columns if _key(c) in accepted), None)


def _long_frame_records(frame: pd.DataFrame) -> list[dict[str, Any]] | None:
    """Преобразует Показатель|Значение|Единица в карточку пациента."""
    name_col = _column_by_header(frame, _NAME_HEADERS)
    value_col = _column_by_header(frame, _VALUE_HEADERS)
    if not name_col or not value_col:
        return None
    unit_col = _column_by_header(frame, _UNIT_HEADERS)
    patient_col = _column_by_header(frame, _PATIENT_HEADERS)

    groups: list[tuple[Any, pd.DataFrame]]
    if patient_col and frame[patient_col].notna().any():
        groups = list(frame.groupby(patient_col, dropna=False, sort=False))
    else:
        groups = [(None, frame)]

    records: list[dict[str, Any]] = []
    for patient_id, group in groups:
        record: dict[str, Any] = {}
        if patient_id is not None and not _is_empty(patient_id):
            record["patient_id"] = str(patient_id)
        for _, row in group.iterrows():
            label = row.get(name_col)
            value = row.get(value_col)
            if _is_empty(label) or _is_empty(value):
                continue
            canonical = field(label)
            if canonical is None:
                continue
            unit = row.get(unit_col, "") if unit_col else ""
            if canonical in LABS:
                record[canonical] = {"value": value, "unit": "" if _is_empty(unit) else str(unit).strip()}
            elif canonical in {"age_years", "sex", "patient_id"}:
                record[canonical] = value
        if record:
            records.append(record)
    return records


def _wide_frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Канонизирует заголовки широкой таблицы, сохраняя неизвестные колонки."""
    records: list[dict[str, Any]] = []
    for source in frame.to_dict("records"):
        record: dict[str, Any] = {}
        units: dict[str, str] = {}
        for key, value in source.items():
            if _is_empty(value):
                continue
            canonical = field(key)
            if canonical:
                record[canonical] = value
                continue
            normalized = _key(key)
            # Колонки вида "Гемоглобин единица" / "hemoglobin_unit".
            for suffix in ("единица", "едизм", "unit", "units"):
                if normalized.endswith(suffix):
                    base = normalized[: -len(suffix)]
                    lab = field(base)
                    if lab in LABS:
                        units[lab] = str(value).strip()
                    break
        if units:
            record["units"] = units
        if record:
            records.append(record)
    return records


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    frame = _prepare_frame(frame)
    long_records = _long_frame_records(frame)
    if long_records is not None:
        return long_records
    return _wide_frame_records(frame)


def _decode_text(data: bytes) -> str:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1251", "cp866", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Не удалось определить кодировку текстового файла: " + "; ".join(errors[-2:]))


def _read_delimited(data: bytes) -> list[dict[str, Any]]:
    text = _decode_text(data)
    last_error: Exception | None = None
    # header=None позволяет найти реальный заголовок после служебной шапки.
    for separator in (None, ";", "\t", ",", "|"):
        try:
            kwargs = {"sep": separator, "header": None, "dtype": object}
            if separator is None:
                kwargs.update({"engine": "python"})
            raw = pd.read_csv(io.StringIO(text), **kwargs)
            if raw.shape[1] < 2:
                continue
            records = _frame_records(raw)
            if records:
                return records
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Не удалось прочитать текстовую таблицу: {last_error}")


def _read_excel(data: bytes) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=object)
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать Excel: {exc}") from exc
    all_records: list[dict[str, Any]] = []
    for _, raw in sheets.items():
        try:
            records = _frame_records(raw)
            if records:
                all_records.extend(records)
        except Exception as exc:
            last_error = exc
    if not all_records:
        raise ValueError(f"В листах Excel не найдены распознаваемые данные: {last_error}")
    return all_records[:MAX_ROWS]


def _read_json(data: bytes) -> list[dict[str, Any]]:
    obj = json.loads(_decode_text(data))
    if isinstance(obj, dict) and "patients" in obj:
        obj = obj["patients"]
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        raise ValueError("JSON должен содержать объект пациента или список пациентов")
    if len(obj) > MAX_ROWS:
        raise ValueError(f"JSON содержит более {MAX_ROWS} пациентов")
    result = [dict(item) for item in obj if isinstance(item, dict)]
    if not result:
        raise ValueError("JSON не содержит объектов пациентов")
    return result


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join(self._cell).strip())
            self._cell = None
        elif tag.lower() == "tr" and self._row is not None:
            if any(cell for cell in self._row):
                self.rows.append(self._row)
            self._row = None


def _read_html(data: bytes) -> list[dict[str, Any]]:
    parser = _TableHTMLParser()
    parser.feed(_decode_text(data))
    if not parser.rows:
        raise ValueError("В HTML не найдены таблицы")
    width = max(len(row) for row in parser.rows)
    raw = pd.DataFrame([row + [None] * (width - len(row)) for row in parser.rows])
    records = _frame_records(raw)
    if not records:
        raise ValueError("В HTML-таблице не найдены лабораторные показатели")
    return records


def _pdf_pages(data: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(data))
    pages = [(page.extract_text() or "").replace("\u00a0", " ") for page in reader.pages]
    if not any(len(page.strip()) >= 30 for page in pages):
        raise ValueError(
            "PDF не содержит текстового слоя (вероятно, это скан/фото). "
            "Без OCR такой файл распознать нельзя; загрузите текстовый PDF, CSV или XLSX."
        )
    return pages


def _clean_inline(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" \t:;-—")


_SECTION_HEADINGS = [
    "ФИО пациента", "Ф. И. О. пациента", "Ф. И. О пациента", "Ф.И.О. пациента", "Ф.И.О пациента", "Ф И О пациента",
    "Фамилия, имя, отчество", "Фамилия Имя Отчество", "Фамилия Имя Отчество пациента",
    "ФИО", "Ф. И. О.", "Ф.И.О.", "Ф И О", "Пациент", "Patient name", "Patient",
    "Фамилия", "Имя", "Отчество",
    "Возраст", "Пол", "Дата рождения",
    "Жалобы", "Жалобы при обращении", "Симптомы", "Клинические симптомы",
    "Анамнез", "Анамнез заболевания", "Анамнез жизни", "Хронические заболевания",
    "Сопутствующие заболевания", "Постоянные препараты", "Постоянно принимаемые препараты", "Препараты",
    "Лекарственные препараты", "Медикаменты", "Принимает", "Принимала", "Принимал",
    "Приём препаратов", "Прием препаратов",
    "Лабораторные показатели", "Результаты исследований", "Результаты анализов",
    "Исследование", "Показатель", "Результат", "Единицы измерения",
    "Референсные значения", "Диагноз", "Заключение", "Рекомендации", "Врач",
    "Лаборатория", "Биоматериал", "Материал", "Метод", "Номер заказа", "Заказ",
    "Дата обращения", "Дата взятия", "Дата регистрации", "Адрес", "Телефон",
    "E-mail", "Email", "СНИЛС", "Полис", "Страховая", "Место работы", "Отделение",
    "Кабинет", "Организация", "Учреждение", "Комментарий", "Интерпретация",
    "Контакты", "Статус",
    "Complaints", "Symptoms", "History", "Medical history", "Chronic conditions",
    "Medications", "Age", "Sex", "Laboratory results", "Results",
]

_ADMIN_LINE_RE = re.compile(
    r"^(?:дата\s+(?:обращения|взятия|регистрации|выдачи|печати)|номер\s+(?:заказа|образца)|"
    r"заказ|лаборатория|врач|лечащий\s+врач|биоматериал|материал|метод|референс(?:ные\s+значения)?|"
    r"штрих[- ]?код|barcode|sample|specimen|reference\s+range|laboratory|"
    r"адрес|телефон|e-?mail|снилс|полис|страховая|место\s+работы|отделение|кабинет|"
    r"организация|учреждение|контакты|статус)\b",
    flags=re.I,
)


def _strip_clinical_noise(line: str) -> str:
    line = str(line or "").replace("\u00a0", " ").strip()
    line = re.sub(r"^[•·▪▫◦●○■□◆◇►▸▶✓✔☑☐*\\-–—→]+\\s*", "", line)
    line = re.sub(
        r"^(?:при\s+обращении|на\s+момент\s+обращения|со\s+слов\s+пациента)\s*[:;,\-–—]*\s*",
        "",
        line,
        flags=re.I,
    )
    line = _clean_inline(line)
    if not line or _ADMIN_LINE_RE.match(line):
        return ""
    return line


def _looks_like_section_heading(line: str) -> bool:
    cleaned = _clean_inline(line)
    if not cleaned:
        return False
    for label in sorted(_SECTION_HEADINGS, key=len, reverse=True):
        if re.match(rf"^{re.escape(label)}\s*[:=\-—]?\s*", cleaned, flags=re.I):
            return True
    return False


def _looks_like_lab_line(line: str) -> bool:
    """Return True only for a line that structurally resembles a lab result.

    This deliberately avoids treating clinical prose such as
    "принимала железо последние 3 месяца" as a serum-iron measurement.
    """
    source = str(line or "").replace("\u00a0", " ")
    low = source.lower().replace("ё", "е")
    for aliases in _PDF_ALIASES.values():
        for alias in sorted(aliases, key=len, reverse=True):
            pat = re.search(
                rf"(?<![a-zа-я0-9]){re.escape(alias.lower().replace('ё','е'))}(?![a-zа-я0-9])",
                low,
            )
            if not pat:
                continue
            tail = source[pat.end() : pat.end() + 90]
            number = re.search(r"[<>]?\s*[-+]?\d+(?:[.,]\d+)?", tail)
            if not number:
                continue
            between = tail[: number.start()]
            after = tail[number.end() : number.end() + 40].replace("*", " ")
            if _UNIT_RE.search(after):
                return True
            # A direct number followed by a duration/dose/form is usually medication
            # prose, not a laboratory result (e.g. "железо 3 месяца", "железо 100 мг").
            if re.match(
                r"\s*(?:мг|мкг|г|таб(?:л(?:еток|етки)?)?|капс(?:ул[аы]?)?|"
                r"раз(?:а|\s+в\s+день)?|дн(?:я|ей)?|недел[яьи]|месяц(?:а|ев)?|год(?:а|ов)?)\b",
                after,
                flags=re.I,
            ):
                continue
            # Unit may be omitted in some exports. In that case accept only a
            # near-direct "analyte -> number" pattern, allowing parenthetical
            # abbreviations but not ordinary words.
            simplified = re.sub(r"\([^)]{0,30}\)", "", between)
            simplified = re.sub(r"[\s:;=,\-–—<>/\\]+", "", simplified)
            if simplified == "":
                return True
    return False


def _clip_inline_heading(value: str) -> str:
    """Stop a clinical field when a second labelled field was flattened onto the same line."""
    text = str(value or "")
    if not text:
        return ""
    labels = sorted(_SECTION_HEADINGS, key=len, reverse=True)
    label_re = "|".join(re.escape(x) for x in labels)
    # Require punctuation/spacing before the next label and an explicit delimiter after it.
    # This avoids cutting ordinary prose that merely contains a word such as "врач".
    m = re.search(rf"(?:[;|]\s*|\s{{2,}})(?:{label_re})\s*[:=\-—]", text, flags=re.I)
    if m:
        text = text[:m.start()]
    return _clean_inline(text)


def _looks_like_structured_free_text(text: str) -> bool:
    """Prefer the free-text parser only when the document actually contains labelled clinical fields."""
    labels = [
        "фио пациента", "ф. и. о. пациента", "ф. и. о пациента", "ф.и.о. пациента", "ф.и.о пациента", "фамилия, имя, отчество", "фамилия имя отчество", "фио", "ф. и. о.", "ф.и.о.", "пациент", "возраст", "пол", "дата рождения",
        "жалобы при обращении", "жалобы", "клинические симптомы", "симптомы",
        "анамнез заболевания", "анамнез жизни", "анамнез", "хронические заболевания",
        "сопутствующие заболевания", "постоянные препараты", "лекарственные препараты", "препараты", "принимает", "принимала", "принимал",
        "patient", "age", "sex",
        "complaints", "symptoms", "history", "medications",
    ]
    sample = str(text or "")[:12000]
    hits = 0
    for label in labels:
        if re.search(rf"(?:^|\n)\s*{re.escape(label)}\s*[:=\-—]", sample, flags=re.I):
            hits += 1
    return hits >= 2


def _heading_matches(text: str) -> list[tuple[int, int, str]]:
    """Return labelled section boundaries, including headings flattened onto one line.

    PDF text extraction often turns a form such as
    "Жалобы: ...  Анамнез: ...  Препараты: ..." into one physical line.
    We therefore locate explicit ``heading:`` markers across the whole text rather
    than relying only on line breaks. Headings without a delimiter are accepted
    only at the beginning of a line.
    """
    source = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    labels = sorted(set(_SECTION_HEADINGS), key=len, reverse=True)
    label_re = "|".join(re.escape(x) for x in labels)

    matches: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()

    # High-confidence form: an explicit delimiter after the heading, anywhere.
    explicit = re.compile(rf"(?<![A-Za-zА-Яа-яЁё0-9])(?P<label>{label_re})\s*[:=—-]\s*", flags=re.I)
    for m in explicit.finditer(source):
        span = (m.start(), m.end())
        if span not in seen:
            matches.append((m.start(), m.end(), m.group("label")))
            seen.add(span)

    # Some exports put a section heading on a dedicated line without a colon.
    # Accept only a heading-only line.  A loose "label + any text" rule breaks
    # phrases such as "препараты железа 3 месяца" inside the anamnesis.
    line_start = re.compile(rf"(?im)^[ \t]*(?P<label>{label_re})[ \t]*(?=\n|$)")
    for m in line_start.finditer(source):
        span = (m.start(), m.end())
        if span not in seen:
            matches.append((m.start(), m.end(), m.group("label")))
            seen.add(span)

    # When a longer label starts at the same position (e.g. "ФИО пациента"
    # and "ФИО"), keep the longest/highest-confidence match.
    by_start: dict[int, tuple[int, int, str]] = {}
    for item in matches:
        current = by_start.get(item[0])
        if current is None or item[1] > current[1]:
            by_start[item[0]] = item
    return [by_start[k] for k in sorted(by_start)]


def _clean_clinical_block(value: str) -> str:
    """Clean one already-bounded clinical block without inventing content."""
    parts: list[str] = []
    for raw in str(value or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        # PDF reports often use long line-drawing separators around headings.
        # They are layout, not clinical text.
        if re.fullmatch(r"[\s─━═_\-]{3,}", str(raw or "")):
            continue
        cleaned = _strip_clinical_noise(raw)
        if not cleaned:
            if parts:
                break
            continue
        if _looks_like_lab_line(cleaned):
            break
        # Drop obvious standalone punctuation/technical remnants from PDF extraction.
        cleaned = re.sub(r"^[|/:;,._]+|[|]+$", "", cleaned).strip()
        if cleaned:
            parts.append(cleaned)
    return _clean_inline(" ".join(parts))[:700]


def _extract_labeled_blocks(text: str, labels: list[str]) -> str:
    """Extract only explicitly labelled clinical sections.

    The next labelled field is a hard boundary even when PDF extraction flattened
    several fields onto a single line. This prevents anamnesis, complaints and
    medications from leaking into one another.
    """
    source = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    targets = {x.lower().replace("ё", "е") for x in labels}
    headings = _heading_matches(source)
    blocks: list[str] = []

    for idx, (start, content_start, label) in enumerate(headings):
        norm = label.lower().replace("ё", "е")
        if norm not in targets:
            continue
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(source)
        raw_value = source[content_start:end]
        value = _clean_clinical_block(raw_value)
        if value and value.lower() not in {x.lower() for x in blocks}:
            blocks.append(value)
    return "\n".join(blocks)[:1200]


def _clean_name_candidate(value: str) -> str:
    candidate = str(value or "").replace("\u00a0", " ")
    # Stop common demographic/administrative continuations on the same PDF line.
    candidate = re.split(
        r"\s+(?=(?:возраст|пол|дата\s+рождения|дата\s+обращения|номер\s+заказа|"
        r"телефон|адрес|снилс|полис|patient\s*id)\s*[:=—-])",
        candidate,
        maxsplit=1,
        flags=re.I,
    )[0]
    candidate = re.sub(r"\s*,?\s*\d{1,3}\s*(?:лет|года?)\b.*$", "", candidate, flags=re.I)
    candidate = _clean_inline(candidate)
    return candidate


def _valid_person_name(candidate: str) -> bool:
    if not candidate:
        return False
    if len(candidate) > 120:
        return False
    tokens = [t for t in re.split(r"\s+", candidate) if t]
    if not 2 <= len(tokens) <= 5:
        return False
    # Permit initials (И., I.) and hyphenated surnames, reject IDs/numeric blobs.
    good = 0
    for token in tokens:
        bare = token.strip(".,")
        if re.fullmatch(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'’-]*", bare) or re.fullmatch(r"[A-Za-zА-Яа-яЁё]", bare):
            good += 1
    return good == len(tokens)


def _extract_single_labeled_value(text: str, labels: list[str]) -> str:
    source = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    targets = {x.lower().replace("ё", "е") for x in labels}
    headings = _heading_matches(source)
    for idx, (_, content_start, label) in enumerate(headings):
        if label.lower().replace("ё", "е") not in targets:
            continue
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(source)
        raw = source[content_start:end]
        first = next((x for x in raw.split("\n") if x.strip()), "")
        return _clean_inline(first)
    return ""


def _extract_full_name(text: str) -> str:
    source = str(text or "").replace("\r\n", "\n").replace("\r", "\n")

    # Flexible Russian F.I.O. spelling: PDF exports vary in dots/spaces and often
    # use "Фамилия, имя, отчество" instead of the abbreviation.
    direct = re.search(
        r"(?im)^[ \t]*(?:"
        r"ф\s*\.?\s*и\s*\.?\s*о\s*\.?(?:\s+пациента)?|"
        r"фио(?:\s+пациента)?|"
        r"фамилия\s*,?\s*имя\s*,?\s*отчество(?:\s+пациента)?|"
        r"patient\s+name"
        r")\s*[:=—-]\s*(.+?)\s*$",
        source,
    )
    if direct:
        candidate = _clean_name_candidate(direct.group(1))
        if _valid_person_name(candidate):
            return candidate

    fio_labels = [
        "ФИО пациента", "Ф. И. О. пациента", "Ф. И. О пациента", "Ф.И.О. пациента", "Ф.И.О пациента", "Ф И О пациента",
        "Фамилия, имя, отчество", "Фамилия Имя Отчество", "Фамилия Имя Отчество пациента",
        "ФИО", "Ф. И. О.", "Ф.И.О.", "Ф И О", "Пациент", "Patient name", "Patient",
    ]
    candidate = _clean_name_candidate(_extract_single_labeled_value(text, fio_labels))
    if _valid_person_name(candidate):
        return candidate

    # Some forms store surname/name/patronymic in separate fields.
    surname = _clean_name_candidate(_extract_single_labeled_value(text, ["Фамилия"]))
    name = _clean_name_candidate(_extract_single_labeled_value(text, ["Имя"]))
    patronymic = _clean_name_candidate(_extract_single_labeled_value(text, ["Отчество"]))
    pieces = [x for x in (surname, name, patronymic) if x]
    candidate = _clean_inline(" ".join(pieces))
    if len(pieces) >= 2 and _valid_person_name(candidate):
        return candidate

    # Final conservative fallback for "ФИО пациента Иванов Иван Иванович" without ':'.
    fallback = re.search(
        r"(?im)^[ \t]*(?:фио(?:\s+пациента)?|ф\.?\s*и\.?\s*о\.?(?:\s+пациента)?|"
        r"пациент|patient(?:\s+name)?)\s+"
        r"([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'’.-]*(?:\s+[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'’.-]*){1,4})\s*$"
    , str(text or ""))
    if fallback:
        candidate = _clean_name_candidate(fallback.group(1))
        if _valid_person_name(candidate):
            return candidate
    return ""


def _extract_demographics(text: str, record: dict[str, Any]) -> None:
    full_name = _extract_full_name(text)
    if full_name:
        record["full_name"] = full_name

    age = re.search(r"(?:возраст|age)\s*[:=\-—]?\s*(\d{1,3})(?:\s*(?:лет|года?))?", text, flags=re.I)
    if age:
        record["age_years"] = int(age.group(1))
    else:
        dob = re.search(
            r"(?:дата\s*рождения|д\.\s*р\.|date\s*of\s*birth|dob)\s*[:=\-—]?\s*(\d{1,2})[./-](\d{1,2})[./-](\d{4})",
            text,
            flags=re.I,
        )
        if dob:
            from datetime import date
            try:
                born = date(int(dob.group(3)), int(dob.group(2)), int(dob.group(1)))
                today = date.today()
                record["age_years"] = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
            except ValueError:
                pass

    sex = re.search(r"(?:пол|sex|gender)\s*[:=\-—]?\s*(жен(?:ский)?|муж(?:ской)?|female|male|[fmжм])\b", text, flags=re.I)
    if sex:
        record["sex"] = sex.group(1)
    patient = re.search(r"(?:инз|код\s+пациента|patient\s*id)\s*[:=\-—]?\s*([A-Za-zА-Яа-я0-9_-]{2,64})", text, flags=re.I)
    if patient:
        record["patient_id"] = patient.group(1)


def _extract_clinical_context(text: str, record: dict[str, Any]) -> None:
    # UX has one field "Жалобы / симптомы". Explicit Symptoms blocks are merged into it.
    complaints = _extract_labeled_blocks(
        text,
        ["жалобы при обращении", "жалобы", "клинические симптомы", "симптомы", "complaints", "symptoms"],
    )
    history = _extract_labeled_blocks(
        text,
        [
            "анамнез заболевания", "анамнез жизни", "анамнез", "хронические заболевания",
            "сопутствующие заболевания", "medical history", "chronic conditions", "history",
        ],
    )
    medications = _extract_labeled_blocks(
        text,
        [
            "постоянные препараты", "постоянно принимаемые препараты", "лекарственные препараты", "медикаменты", "препараты",
            "принимает", "принимала", "принимал", "приём препаратов", "прием препаратов", "medications",
        ],
    )
    if complaints:
        record["complaints"] = complaints
    if history:
        record["history"] = history
    if medications:
        record["medications"] = medications


_CLINICAL_CONTEXT_HEADINGS = {
    "жалобы при обращении", "жалобы", "клинические симптомы", "симптомы",
    "complaints", "symptoms",
    "анамнез заболевания", "анамнез жизни", "анамнез", "хронические заболевания",
    "сопутствующие заболевания", "medical history", "chronic conditions", "history",
    "постоянные препараты", "постоянно принимаемые препараты", "лекарственные препараты", "медикаменты", "препараты",
    "принимает", "принимала", "принимал", "приём препаратов", "прием препаратов", "medications",
}


def _position_is_inside_clinical_context(text: str, position: int) -> bool:
    """Do not interpret words in complaints/history/medications as lab analytes."""
    headings = _heading_matches(text)
    active: str | None = None
    for idx, (start, _content_start, label) in enumerate(headings):
        if start > position:
            break
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        if start <= position < end:
            active = label.lower().replace("ё", "е")
            break
    return active in {x.lower().replace("ё", "е") for x in _CLINICAL_CONTEXT_HEADINGS}


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![A-Za-zА-Яа-я0-9]){escaped}(?![A-Za-zА-Яа-я0-9])", flags=re.I)


def _extract_lab_candidates_from_text(text: str, key: str, aliases: list[str]) -> list[dict[str, str]]:
    """Return all plausible measurements for one analyte in a text block.

    This is deliberately conservative: occurrences inside clinical prose are ignored
    unless the line structurally resembles a laboratory result.  Multiple distinct
    values are surfaced as a conflict instead of silently choosing one.
    """
    matches: list[re.Match[str]] = []
    for alias in sorted(aliases, key=len, reverse=True):
        matches.extend(_alias_pattern(alias).finditer(text))
    matches.sort(key=lambda m: m.start())
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in matches:
        # Ignore aliases that occur inside a parenthetical qualifier of another
        # analyte, e.g. "Магний (эритроциты)" must not become an RBC result.
        line_start = text.rfind("\n", 0, match.start()) + 1
        prefix_on_line = text[line_start:match.start()]
        if prefix_on_line.count("(") > prefix_on_line.count(")"):
            continue
        if _position_is_inside_clinical_context(text, match.start()):
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            if line_end < 0:
                line_end = len(text)
            if not _looks_like_lab_line(text[line_start:line_end]):
                continue
        tail = text[match.end() : match.end() + 160]
        number = re.search(r"[<>]?\s*[-+]?\d+(?:[.,]\d+)?", tail)
        if not number:
            continue
        prefix = tail[: number.start()].lower()
        if any(stop in prefix for stop in ("дата ", "тест №", "результат\nединицы")):
            continue
        value = re.sub(r"\s+", "", number.group(0))
        after = tail[number.end() : number.end() + 50].replace("*", " ")
        unit_match = _UNIT_RE.search(after)
        unit = unit_match.group(0).strip() if unit_match else ""
        pair = (value.replace(",", "."), unit.lower())
        if pair not in seen:
            seen.add(pair)
            out.append({"value": value, "unit": unit})
    return out


def _extract_lab_from_text(text: str, key: str, aliases: list[str]) -> dict[str, str] | None:
    candidates = _extract_lab_candidates_from_text(text, key, aliases)
    return candidates[0] if candidates else None


def _merge_pdf_records(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Merge pages of one report into one editable patient card."""
    warnings: list[str] = []
    patient_ids = {str(r.get("patient_id")).strip() for r in records if r.get("patient_id")}
    if len(patient_ids) > 1:
        raise ValueError("В PDF обнаружены разные идентификаторы пациентов. Загрузите файл только одного пациента.")

    merged: dict[str, Any] = {}
    if patient_ids:
        merged["patient_id"] = next(iter(patient_ids))

    for demo in ("full_name", "age_years", "sex"):
        values = [r.get(demo) for r in records if r.get(demo) not in (None, "")]
        if values:
            normalized = {str(v).strip().lower() for v in values}
            if len(normalized) > 1 and demo == "full_name":
                raise ValueError("В PDF обнаружены разные ФИО. Загрузите документ только одного пациента.")
            if len(normalized) > 1:
                warnings.append(f"В разных страницах PDF расходится поле {demo}; оставлено первое значение, проверьте вручную.")
            merged[demo] = values[0]

    # Clinical text can legitimately continue on several pages, so preserve unique fragments.
    for key in ("complaints", "history", "medications"):
        values: list[str] = []
        for r in records:
            value = _clean_inline(r.get(key, ""))
            if value and value.lower() not in {x.lower() for x in values}:
                values.append(value)
        if values:
            merged[key] = "\n".join(values)[:1600]

    conflicts: list[str] = []
    for record in records:
        conflicts.extend(record.get("_lab_conflicts", []) if isinstance(record.get("_lab_conflicts"), list) else [])
        for key, value in record.items():
            if key not in LABS:
                continue
            if key not in merged:
                merged[key] = value
                continue
            if json.dumps(merged[key], ensure_ascii=False, sort_keys=True, default=str) != json.dumps(value, ensure_ascii=False, sort_keys=True, default=str):
                conflicts.append(LABS[key]["label"])
    if conflicts:
        unique_conflicts = sorted(set(conflicts))
        # A conflicting duplicate is unsafe to auto-select.  Keep the rest of the
        # patient card, but leave this analyte blank for physician verification.
        labels_to_keys = {LABS[k]["label"]: k for k in LABS}
        for label in unique_conflicts:
            key = labels_to_keys.get(label)
            if key:
                merged.pop(key, None)
        merged["_lab_conflicts"] = unique_conflicts
        warnings.append(
            "На нескольких страницах найдены дубликаты с разными значениями: "
            + ", ".join(unique_conflicts)
            + ". Конфликтующие показатели не подставлены автоматически."
        )
    return merged, warnings


def _read_pdf(data: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    page_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for page_number, text in enumerate(_pdf_pages(data), start=1):
        if len(text.strip()) < 30:
            continue
        record: dict[str, Any] = {}
        _extract_demographics(text, record)
        _extract_clinical_context(text, record)
        page_conflicts: list[str] = []
        for key, aliases in _PDF_ALIASES.items():
            candidates = _extract_lab_candidates_from_text(text, key, aliases)
            if len(candidates) > 1:
                distinct = {str(x.get("value", "")).replace(",", ".") for x in candidates}
                if len(distinct) > 1:
                    page_conflicts.append(LABS[key]["label"])
                    continue
            if candidates:
                record[key] = candidates[0]
        if page_conflicts:
            record["_lab_conflicts"] = page_conflicts
        # Keep context-only pages as well: patient card may be on page 1 and labs on page 2.
        if record:
            record["_source_page"] = page_number
            page_records.append(record)

    if not page_records:
        raise ValueError("В текстовом PDF не удалось распознать данные пациента или лабораторные показатели")

    merged, merge_warnings = _merge_pdf_records(page_records)
    if not any(key in merged for key in LABS):
        raise ValueError("В текстовом PDF не удалось распознать лабораторные показатели")
    warnings.extend(merge_warnings)
    lab_pages = sum(1 for r in page_records if any(k in r for k in LABS))
    if lab_pages > 1:
        warnings.append(f"Распознано страниц с анализами: {lab_pages}. Они объединены в одну карточку пациента; проверьте результат перед расчётом.")
    if any(k in merged for k in ("full_name", "age_years", "sex", "complaints", "history", "medications")):
        warnings.append("Из PDF также предзаполнена карточка пациента. Проверьте ФИО, возраст, пол, жалобы, анамнез и препараты перед расчётом.")
    return [merged], warnings


def _read_free_text(data: bytes) -> list[dict[str, Any]]:
    """Parse a plain TXT/clinical export that is not a delimited table."""
    text = _decode_text(data).replace("\u00a0", " ")
    if len(text.strip()) < 10:
        raise ValueError("Текстовый файл не содержит достаточно данных")

    # Explicitly reject several distinct labelled patients in one free-text document.
    fio_labels = {
        "фио пациента", "ф. и. о. пациента", "ф. и. о пациента", "ф.и.о. пациента", "ф.и.о пациента", "ф и о пациента",
        "фамилия, имя, отчество", "фамилия имя отчество", "фамилия имя отчество пациента",
        "фио", "ф. и. о.", "ф.и.о.", "ф и о", "пациент", "patient name", "patient",
    }
    headings = _heading_matches(text)
    names: list[str] = []
    for idx, (_, content_start, label) in enumerate(headings):
        if label.lower().replace("ё", "е") not in fio_labels:
            continue
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
        raw = text[content_start:end]
        first = next((x for x in raw.split("\n") if x.strip()), "")
        name = _clean_name_candidate(first)
        if _valid_person_name(name):
            names.append(name)
    if len({x.lower() for x in names}) > 1:
        raise ValueError("В TXT обнаружены разные ФИО. Загрузите файл только одного пациента.")

    record: dict[str, Any] = {}
    _extract_demographics(text, record)
    _extract_clinical_context(text, record)
    conflicts: list[str] = []
    for key, aliases in _PDF_ALIASES.items():
        candidates = _extract_lab_candidates_from_text(text, key, aliases)
        if len(candidates) > 1:
            distinct = {str(x.get("value", "")).replace(",", ".") for x in candidates}
            if len(distinct) > 1:
                conflicts.append(LABS[key]["label"])
                continue
        if candidates:
            record[key] = candidates[0]
    if conflicts:
        record["_lab_conflicts"] = conflicts
    if not any(key in record for key in LABS):
        raise ValueError("В свободном тексте не удалось распознать лабораторные показатели")
    return [record]


def _looks_like_pdf(data: bytes) -> bool:
    return data.lstrip().startswith(b"%PDF")


def _looks_like_excel(data: bytes) -> bool:
    # XLSX — ZIP-контейнер; старый XLS — Compound File Binary.
    return data.startswith(b"PK\x03\x04") or data.startswith(bytes.fromhex("D0CF11E0A1B11AE1"))


def _looks_like_json(data: bytes) -> bool:
    return data.lstrip()[:1] in {b"{", b"["}


def _looks_like_html(data: bytes) -> bool:
    head = data[:4096].lower()
    return b"<html" in head or b"<table" in head



def _single_patient(records: list[dict[str, Any]], warnings: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    if not records:
        raise ValueError("В файле не найдено данных пациента")
    if len(records) > 1:
        raise ValueError(
            f"В файле распознано несколько пациентов/строк ({len(records)}). "
            "Текущий экран предназначен для одного пациента: оставьте в файле одну карточку или загрузите индивидуальный бланк."
        )
    record = records[0]
    conflicts = record.pop("_lab_conflicts", []) if isinstance(record, dict) else []
    if conflicts:
        warnings.append(
            "Найдены дубликаты показателей с разными значениями: "
            + ", ".join(sorted(set(str(x) for x in conflicts)))
            + ". Конфликтующие показатели не подставлены автоматически. Проверьте исходный файл и заполните их вручную."
        )
    warnings.append("Распознанные значения необходимо проверить перед расчётом.")
    return records, warnings

def read_upload(data: bytes, filename: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Read a single-patient laboratory file.

    The UI no longer supports batch processing.  Multi-row files are rejected instead
    of being silently interpreted as several patients.  PDF pages of the same report
    are merged by _read_pdf and remain manually editable before screening.
    """
    _check_size(data)
    suffix = Path(filename or "").suffix.lower()
    warnings: list[str] = []

    if _looks_like_pdf(data):
        records, pdf_warnings = _read_pdf(data)
        return _single_patient(records, pdf_warnings)
    if _looks_like_excel(data):
        return _single_patient(_read_excel(data), warnings)
    if _looks_like_json(data):
        return _single_patient(_read_json(data), warnings)
    if _looks_like_html(data):
        return _single_patient(_read_html(data), warnings)

    preferred: list[Callable[[bytes], list[dict[str, Any]]]] = []
    if suffix in {".xlsx", ".xls"}:
        preferred.append(_read_excel)
    elif suffix == ".json":
        preferred.append(_read_json)
    elif suffix in {".html", ".htm"}:
        preferred.append(_read_html)
    elif suffix == ".txt":
        # TXT can be either a delimited export or an ordinary clinical/lab text.
        # Prefer the clinical parser only when explicit labelled fields are present;
        # otherwise preserve normal TSV/CSV-in-TXT behaviour.
        decoded = _decode_text(data)
        if _looks_like_structured_free_text(decoded):
            preferred.extend([_read_free_text, _read_delimited])
        else:
            preferred.extend([_read_delimited, _read_free_text])
    preferred.extend([_read_delimited, _read_free_text, _read_json, _read_html, _read_excel])

    errors: list[str] = []
    tried: set[Callable] = set()
    for parser in preferred:
        if parser in tried:
            continue
        tried.add(parser)
        try:
            records = parser(data)
            if records:
                if suffix not in {".csv", ".tsv", ".txt", ".json", ".xlsx", ".xls", ".html", ".htm"}:
                    warnings.append("Формат определён по содержимому файла, а не по расширению.")
                return _single_patient(records, warnings)
        except Exception as exc:
            # A parser that successfully identified several patients must not be
            # silently bypassed by a looser fallback parser.
            if isinstance(exc, ValueError) and "несколько пациентов" in str(exc).lower():
                raise
            errors.append(f"{parser.__name__}: {exc}")

    raise ValueError(
        "Не удалось определить структуру файла. Поддерживаются текстовые PDF, JSON, "
        "CSV/TSV/TXT, XLSX/XLS и HTML-таблицы. Детали: " + " | ".join(errors[:4])
    )
