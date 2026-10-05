"""Streamlit-интерфейс врача «Мидия»: ввод/импорт анализов, скрининг, отчёты.

Вся медицинская логика находится в пакете ``screening``; этот файл отвечает только
за отображение и работу с состоянием сессии.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from screening.casebase import add_case, delete_case, list_cases
from screening.coding import ICD10_BY_CLASS
from screening.communication import build_patient_conversation
from screening.context_ai import structure_clinical_context
from screening.importers import read_upload
from screening.rag import BOOKS_DIR, generate_ai_explanation, ollama_status, refresh_knowledge_cache
from screening.report import generate_patient_pdf, generate_pdf
from screening.schema import CLASSES, DEFAULT_LABS, LABS, SYMPTOM_LABELS, UNIT_NOTE, convert_value, field
from screening.service import screen
from screening.similar import local_confirmed_neighbors
from screening.sources import all_sources

# Streamlit требует вызова set_page_config до любого другого вызова st.*.
st.set_page_config(page_title="Мидия", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")

ROOT = Path(__file__).resolve().parent
ASSET_LOGO = ROOT / "assets" / "midia_logo.png"
ASSET_LOGO_FALLBACK = ROOT / "assets" / "sechenov_logo.png"
UI_DISCLAIMER = (
    "«Мидия» — исследовательский прототип врачебной СППР и не зарегистрированное медицинское изделие. "
    "Сервис предназначен только для медицинских работников: результат является рабочей гипотезой, а не диагнозом, "
    "и не используется для автоматического назначения лечения. Модель обучена на частично синтетическом наборе "
    "из 840 наблюдений и 12 классов; метрики внутренние, внешняя клиническая валидация не проводилась, "
    "а Model score не является вероятностью диагноза. Область применения — взрослые небеременные пациенты "
    "в плановом скрининге; перед расчётом врач обязан проверить значения и единицы. Подробности — в разделе «О проекте»."
)


def _default_rows() -> list[dict[str, str]]:
    return [{"Показатель": LABS[k]["label"], "Значение": "", "Единица": LABS[k]["unit"]} for k in DEFAULT_LABS]


def _reset() -> None:
    dark = bool(st.session_state.get("dark_theme", False))
    for key in list(st.session_state):
        del st.session_state[key]
    st.session_state.update({
        "rows": _default_rows(),
        "revision": 0,
        "full_name": "",
        "age": "",
        "sex": "Не указан",
        "complaints": "",
        "history_text": "",
        "medications": "",
        "input_mode": "Загрузить файл",
        "authorization": False,
        "dark_theme": dark,
        "use_ai_import": False,
        "patient_upload_revision": 0,
        "tool_book_upload_revision": 0,
    })


def _ensure_state() -> None:
    if "rows" not in st.session_state:
        _reset()


def _clear_patient_upload() -> None:
    st.session_state["patient_upload_revision"] = int(st.session_state.get("patient_upload_revision", 0)) + 1
    for key in ("import_ok", "import_messages", "ai_import_message", "pending_import_record"):
        st.session_state.pop(key, None)


def _clear_tool_book_upload() -> None:
    st.session_state["tool_book_upload_revision"] = int(st.session_state.get("tool_book_upload_revision", 0)) + 1


def _merge_text(existing: str, incoming: str) -> str:
    existing = str(existing or "").strip()
    incoming = str(incoming or "").strip()
    if not incoming:
        return existing
    if not existing:
        return incoming
    if incoming.lower() in existing.lower():
        return existing
    return existing + "\n" + incoming


def _symptoms_from_text(text: str) -> list[str]:
    low = str(text or "").lower().replace("ё", "е")
    if not low:
        return []
    aliases = {
        "Утомляемость / слабость": ["утом", "слабост", "астени"],
        "Снижение переносимости физической нагрузки": ["переносимост", "нагрузк"],
        "Одышка при нагрузке": ["одыш"],
        "Головокружение": ["головокруж"],
        "Предобморочное состояние / обморок": ["обмор", "предобмор", "синкоп"],
        "Сердцебиение": ["сердцеби", "тахикард"],
        "Головная боль": ["головн", "цефалг"],
        "Бледность кожи / слизистых": ["бледн"],
        "Зябкость / непереносимость холода": ["зяб", "холод"],
        "Извращение вкуса / пика": ["пика", "мел", "лед", "лёд", "извращение вкуса"],
        "Выпадение волос / ломкость ногтей": ["выпадение волос", "ломк", "ногт"],
        "Синдром беспокойных ног": ["беспокойных ног"],
        "Парестезии / покалывание": ["парестез", "покалыв"],
        "Онемение / снижение чувствительности": ["онемен", "чувствительност"],
        "Нарушение походки / равновесия": ["походк", "равновес"],
        "Снижение концентрации / памяти": ["концентрац", "памят"],
        "Глоссит / жжение языка": ["глоссит", "жжение языка"],
        "Стоматит / изменения слизистой полости рта": ["стоматит"],
        "Снижение аппетита": ["аппетит"],
        "Непреднамеренная потеря веса": ["потеря веса", "снижение веса", "похуд"],
        "Хроническая диарея": ["диар", "жидкий стул"],
        "Боль / дискомфорт в животе": ["боль в живот", "дискомфорт в живот"],
        "Чёрный стул / мелена": ["мелена", "черный стул", "чёрный стул"],
        "Кровь в стуле": ["кровь в стуле"],
        "Обильные / длительные менструации": ["обильн", "менорраг", "длительные менстру"],
        "Другие признаки кровопотери": ["кровопот", "кровотеч"],
        "Длительная лихорадка / субфебрилитет": ["лихорад", "субфеб"],
        "Суставные боли / признаки хронического воспаления": ["сустав", "артралг"],
    }
    options = set(SYMPTOM_LABELS.values())
    return [label for label, terms in aliases.items() if label in options and any(term in low for term in terms)]


def _safe_book_name(name: str) -> str:
    base = Path(name or "source.pdf").name
    stem = re.sub(r"[^A-Za-zА-Яа-я0-9._ -]+", "_", base).strip(" .")
    return stem if stem.lower().endswith(".pdf") else stem + ".pdf"


def _apply_import(record: dict) -> None:
    """Перенести распознанную из файла карточку в поля формы (требует проверки врачом)."""
    merged = dict(record)
    if isinstance(record.get("labs"), dict):
        merged.update(record["labs"])
    units = record.get("units", {}) if isinstance(record.get("units"), dict) else {}
    rows: list[dict[str, str]] = []
    for key, value in merged.items():
        canonical = field(key)
        if canonical not in LABS:
            continue
        if isinstance(value, dict):
            raw_value = value.get("value", "")
            unit = value.get("unit", "")
        else:
            raw_value = value
            unit = units.get(key, LABS[canonical]["unit"])
        if raw_value is None or str(raw_value).strip().lower() in {"", "nan", "none"}:
            continue
        rows.append({"Показатель": LABS[canonical]["label"], "Значение": str(raw_value), "Единица": unit or LABS[canonical]["unit"]})
    st.session_state["rows"] = rows or _default_rows()
    st.session_state["revision"] = st.session_state.get("revision", 0) + 1

    parsed_name = str(record.get("full_name", "") or "").strip()
    if parsed_name and not str(st.session_state.get("full_name", "")).strip():
        st.session_state["full_name"] = parsed_name
    if record.get("age_years", record.get("Возраст")) not in (None, ""):
        st.session_state["age"] = str(record.get("age_years", record.get("Возраст", "")))
    sex = str(record.get("sex", record.get("Пол", ""))).upper()
    if sex in {"F", "Ж", "ЖЕНСКИЙ", "FEMALE"}:
        st.session_state["sex"] = "Женский"
    elif sex in {"M", "М", "МУЖСКОЙ", "MALE"}:
        st.session_state["sex"] = "Мужской"

    st.session_state["complaints"] = _merge_text(st.session_state.get("complaints", ""), str(record.get("complaints", "") or ""))
    st.session_state["history_text"] = _merge_text(st.session_state.get("history_text", ""), str(record.get("history", "") or ""))
    st.session_state["medications"] = _merge_text(st.session_state.get("medications", ""), str(record.get("medications", "") or ""))


def _raw_from_form(rows: pd.DataFrame, units_confirmed: bool) -> tuple[dict, list[str]]:
    """Собрать входную запись для screen() из формы; вернуть её и список ошибок ввода."""
    sex = {"Женский": "F", "Мужской": "M"}.get(st.session_state.get("sex"), "")
    raw = {
        "patient_id": "anonymous",
        "age_years": st.session_state.get("age", ""),
        "sex": sex,
        "pregnant": False,
        "complaints": st.session_state.get("complaints", ""),
        "symptoms": _symptoms_from_text(st.session_state.get("complaints", "")),
        "history": st.session_state.get("history_text", ""),
        "medications": st.session_state.get("medications", ""),
        "labs": {},
    }
    errors: list[str] = []
    for row_num, row in enumerate(rows.fillna("").to_dict("records"), 1):
        label = str(row.get("Показатель", "")).strip()
        value = str(row.get("Значение", "")).strip()
        unit = str(row.get("Единица", "")).strip()
        if not value:
            continue
        canonical = field(label)
        if canonical not in LABS:
            errors.append(f"Строка {row_num}: неизвестный показатель «{label}»")
            continue
        if canonical in raw["labs"]:
            errors.append(f"Строка {row_num}: показатель «{LABS[canonical]['label']}» указан повторно")
            continue
        raw["labs"][canonical] = {"value": value, "unit": unit}
        normalized, canonical_unit, _ = convert_value(canonical, value, unit, units_confirmed=units_confirmed)
        maximum = LABS[canonical].get("max")
        if normalized is not None and (normalized < 0 or (maximum is not None and normalized > float(maximum))):
            errors.append(f"Строка {row_num}: {LABS[canonical]['label']} = {normalized:g} вне технического диапазона 0–{float(maximum):g} {canonical_unit}. Проверьте значение и единицу.")
    return raw, errors


def _fingerprint(raw: dict, units_confirmed: bool) -> str:
    """Отпечаток введённых данных: при их изменении прежний результат считается устаревшим."""
    return hashlib.sha256(json.dumps([raw, units_confirmed], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _suggest_units(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Предложить единицы измерения для строк без единицы; врач обязан их подтвердить."""
    out = frame.copy()
    changed = False
    for idx, row in out.fillna("").iterrows():
        canonical = field(str(row.get("Показатель", "")).strip())
        if canonical not in LABS:
            continue
        unit = str(row.get("Единица", "")).strip()
        expected = str(LABS[canonical].get("unit", ""))
        supported = False
        if unit:
            try:
                value, _, _ = convert_value(canonical, 1, unit, units_confirmed=True)
                supported = value is not None
            except Exception:
                supported = False
        if not unit or not supported:
            out.at[idx, "Единица"] = expected
            changed = True
    return out, changed


def _pretty_code(code: str | None) -> str:
    if not code:
        return "—"
    return str(code).replace("_", " ").capitalize()


def _safe_html(value: object) -> str:
    """Экранировать текст перед вставкой в HTML (защита от инъекций в разметку)."""
    return html.escape(str(value if value is not None else ""))


def _logo_data() -> str:
    for asset_path in (ASSET_LOGO, ASSET_LOGO_FALLBACK):
        if asset_path.exists():
            return base64.b64encode(asset_path.read_bytes()).decode("ascii")
    return ""


def _inject_css(dark: bool) -> None:
    if dark:
        bg, card, text, muted, border, pale, ink, input_bg = "#0B1026", "#11183A", "#FFFFFF", "#AEB6DA", "#2C3678", "#18225A", "#7A85DE", "#0F1634"
    else:
        bg, card, text, muted, border, pale, ink, input_bg = "#F7F8FC", "#FFFFFF", "#000000", "#5D6274", "#D8DCEF", "#EEF0FB", "#323FA6", "#FFFFFF"
    st.markdown(f"""
<style>
 :root{{--mid-bg:{bg};--mid-card:{card};--mid-text:{text};--mid-muted:{muted};--mid-border:{border};--mid-pale:{pale};--mid-ink:{ink};--mid-input:{input_bg};--mid-accent:#7A85DE;--mid-blue:#323FA6;--mid-blue-deep:#21108E;--mid-navy:#130758;--mid-link:#0E4D99;--mid-coral:#E85A62;--mid-coral-deep:#D43D58;--mid-coral-soft:rgba(232,90,98,.10);}}
html,body,[class*="css"]{{font-family:Inter,"Segoe UI",Arial,sans-serif;color:var(--mid-text);}}
.stApp{{background:var(--mid-bg);}}
[data-testid="stHeader"]{{background:transparent;}}
.block-container{{max-width:1380px;padding-top:1.2rem;padding-bottom:4rem;}}
h1,h2,h3,h4,h5,h6{{color:var(--mid-text);letter-spacing:-.025em;}}
p,li,label{{color:var(--mid-text);}}
[data-testid="stCaptionContainer"],.stCaption{{color:var(--mid-muted)!important;}}
.midia-header{{display:grid;grid-template-columns:minmax(92px,126px) 1fr auto;gap:20px;align-items:center;background:linear-gradient(180deg,rgba(255,255,255,.72),rgba(238,240,251,.92));border:1px solid rgba(122,133,222,.18);border-radius:0 0 30px 30px;padding:18px 24px 18px 20px;margin:-1.2rem 0 20px;box-shadow:0 8px 28px rgba(50,63,166,.10);}}
.midia-logo-wrap{{display:flex;align-items:center;justify-content:center;align-self:stretch;padding:2px 0;}}
.midia-logo-wrap img{{width:100%;max-width:112px;height:auto;display:block;filter:drop-shadow(0 10px 22px rgba(50,63,166,.12));}}
.midia-title{{text-align:center;color:var(--mid-ink);font-size:clamp(28px,3vw,48px);font-weight:650;line-height:1.04;}}
.midia-sub{{text-align:center;color:var(--mid-ink);font-size:14px;margin-top:6px;opacity:.84;}}
.midia-slogan{{text-align:center;color:var(--mid-blue);font-size:15px;line-height:1.45;margin-top:9px;font-weight:600;letter-spacing:-.01em;max-width:720px;margin-left:auto;margin-right:auto;}}
.midia-hack{{text-align:center;color:var(--mid-muted);font-size:12px;margin-top:6px;}}
.midia-badge{{background:rgba(255,255,255,.78);color:var(--mid-blue);padding:9px 14px;border-radius:999px;font-weight:600;white-space:nowrap;border:1px solid rgba(122,133,222,.16);}}
.midia-import-note{{font-size:14px;color:var(--mid-muted);margin:0 0 8px;}}
.midia-import-note strong{{color:var(--mid-ink);font-weight:600;}}
.midia-doctor-inner-gap{{height:12px;}}
.midia-doctor-section-gap{{height:18px;}}
.midia-disclaimer{{border:1px solid rgba(212,61,88,.28);background:var(--mid-coral-soft);border-left:4px solid var(--mid-coral);border-radius:16px;padding:14px 16px;color:var(--mid-text);margin:12px 0 18px;box-shadow:0 6px 22px rgba(212,61,88,.06);}}
.midia-summary{{background:var(--mid-card);border:1px solid var(--mid-border);border-radius:24px;padding:22px 24px;margin:12px 0 18px;box-shadow:0 8px 28px rgba(50,63,166,.07);}}
.midia-summary h3{{margin:0 0 6px;font-size:22px;color:var(--mid-ink);}}
.midia-code{{font-size:13px;color:var(--mid-muted);}}
.midia-fact-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin-top:14px;}}
.midia-fact{{border:1px solid var(--mid-border);border-radius:14px;padding:11px 13px;background:var(--mid-bg);overflow-wrap:anywhere;}}
.midia-fact strong{{display:block;color:var(--mid-text);font-size:14px;}}
.midia-fact span{{color:var(--mid-muted);font-size:12px;}}
.midia-fact .abn{{color:var(--mid-coral-deep);font-weight:600;}}
.midia-bar-row{{display:grid;grid-template-columns:minmax(200px,1fr) minmax(180px,3fr) 64px;gap:14px;align-items:center;margin:12px 0;}}
.midia-bar-label{{font-weight:600;color:var(--mid-text);font-size:14px;overflow-wrap:anywhere;}}
.midia-bar-track{{height:14px;background:{'#25306A' if dark else '#E7EAF7'};border-radius:999px;overflow:hidden;}}
.midia-bar-fill{{height:100%;background:linear-gradient(90deg,#7A85DE,#323FA6);border-radius:999px;}}
.midia-bar-value{{text-align:right;font-variant-numeric:tabular-nums;color:var(--mid-blue-deep);font-weight:700;}}
.midia-chain{{display:grid;grid-template-columns:1.1fr auto 1fr auto 1fr auto 1.1fr;gap:10px;align-items:stretch;margin:12px 0 18px;}}
.midia-chain-card{{background:var(--mid-card);border:1px solid var(--mid-border);border-radius:16px;padding:14px;min-width:0;overflow-wrap:anywhere;}}
.midia-chain-card b{{color:var(--mid-ink);display:block;margin-bottom:8px;}}
.midia-arrow{{display:flex;align-items:center;color:var(--mid-accent);font-size:24px;font-weight:700;}}
.midia-note{{background:var(--mid-card);border:1px solid var(--mid-border);border-radius:16px;padding:14px 16px;margin:10px 0;overflow-wrap:anywhere;}}
.midia-help{{display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;margin-left:5px;border:1px solid var(--mid-muted);border-radius:50%;font-size:12px;font-weight:700;color:var(--mid-muted);cursor:help;vertical-align:1px;}}
.midia-ref-grid{{display:grid;grid-template-columns:1fr;gap:5px;margin:6px 0 10px;}}
.midia-ref-row{{display:grid;grid-template-columns:minmax(150px,.8fr) minmax(240px,1.7fr);gap:10px;padding:7px 9px;border-bottom:1px solid var(--mid-border);font-size:13px;}}
.midia-ref-code{{color:var(--mid-muted);overflow-wrap:anywhere;}}
.midia-ref-label{{color:var(--mid-text);font-weight:600;overflow-wrap:anywhere;}}
.clinical-finding{{padding:4px 0 6px;color:var(--mid-text);line-height:1.45;}}
.clinical-finding.warning{{color:var(--mid-coral-deep);font-weight:600;text-decoration:underline;text-decoration-color:rgba(212,61,88,.42);text-underline-offset:3px;}}
[data-testid="stForm"],[data-testid="stExpander"]{{border-color:var(--mid-border)!important;border-radius:16px!important;}}
[data-testid="stTextInputRootElement"] input,[data-testid="stTextArea"] textarea,[data-baseweb="select"]>div,[data-testid="stNumberInput"] input{{background:var(--mid-input)!important;color:var(--mid-text)!important;border-color:var(--mid-border)!important;border-radius:10px!important;}}
[data-testid="stTextInputRootElement"] input::placeholder,[data-testid="stTextArea"] textarea::placeholder{{color:var(--mid-muted)!important;opacity:.55;}}
.stButton>button,[data-testid="baseButton-primary"],[data-testid="baseButton-secondary"]{{border-radius:999px!important;min-height:42px;font-weight:600!important;}}
[data-testid="baseButton-primary"]{{background:#323FA6!important;color:#fff!important;border-color:#323FA6!important;}}
[data-testid="baseButton-secondary"]{{background:var(--mid-card)!important;color:var(--mid-ink)!important;border-color:var(--mid-border)!important;}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)){{gap:0;border-bottom:none;overflow-x:auto;padding:4px 0 8px;background:transparent;}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"]{{position:relative;min-height:46px;background:#F1F4FB!important;border:1px solid #D8E1F0!important;border-radius:0!important;padding:11px 18px 11px 46px!important;color:#6E7B91!important;font-weight:600!important;white-space:nowrap;clip-path:polygon(0 0,calc(100% - 18px) 0,100% 50%,calc(100% - 18px) 100%,0 100%,18px 50%);margin:0 -1px 0 0;box-shadow:none!important;}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"] p{{font-size:14px!important;line-height:1.15!important;}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"]:first-child{{clip-path:polygon(0 0,calc(100% - 18px) 0,100% 50%,calc(100% - 18px) 100%,0 100%);border-radius:14px 0 0 14px!important;padding-left:42px!important;}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"]:last-child{{clip-path:polygon(0 0,100% 0,100% 100%,0 100%,18px 50%);border-radius:0 14px 14px 0!important;padding-right:18px!important;}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"][aria-selected="true"]{{background:linear-gradient(180deg,#5067F5,#394BC8)!important;border-color:#394BC8!important;color:#FFFFFF!important;z-index:2;}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"]:hover{{background:#E7EDF8!important;color:#44516A!important;}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"][aria-selected="true"]:hover{{background:linear-gradient(180deg,#5067F5,#394BC8)!important;color:#FFFFFF!important;}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"]::before{{position:absolute;left:14px;top:50%;transform:translateY(-50%);width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;background:#FFFFFF;color:#6B7890;border:1px solid #CDD7E8;}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"][aria-selected="true"]::before{{background:rgba(255,255,255,.20);color:#FFFFFF;border-color:rgba(255,255,255,.35);}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"]:nth-child(1)::before{{content:"1";}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"]:nth-child(2)::before{{content:"2";}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"]:nth-child(3)::before{{content:"3";}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"]:nth-child(4)::before{{content:"4";}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"]:nth-child(5)::before{{content:"5";}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"]:nth-child(6)::before{{content:"6";}}
[data-testid="stTabs"] [data-baseweb="tab-list"]:has(> button[role="tab"]:nth-child(7)) button[role="tab"]:nth-child(7)::before{{content:"7";}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{{display:none!important;}}
[data-testid="stDataFrame"]{{border:1px solid var(--mid-border);border-radius:14px;overflow:hidden;}}
[data-testid="stAlert"]{{border-radius:16px;}}
hr{{border-color:var(--mid-border)!important;}}
@media(max-width:900px){{.midia-header{{grid-template-columns:1fr;text-align:center;padding:18px 16px}}.midia-logo-wrap{{padding-top:4px}}.midia-logo-wrap img{{margin:auto;max-width:96px}}.midia-badge{{justify-self:center}}.midia-chain{{grid-template-columns:1fr}}.midia-arrow{{justify-content:center;transform:rotate(90deg)}}.midia-bar-row{{grid-template-columns:1fr 56px}}.midia-bar-track{{grid-column:1/-1;grid-row:2}}}}
</style>
""", unsafe_allow_html=True)


def _render_header() -> None:
    logo = _logo_data()
    logo_html = f'<div class="midia-logo-wrap"><img src="data:image/png;base64,{logo}" alt="Логотип Мидия"></div>' if logo else '<div></div>'
    st.markdown(
        f'''<div class="midia-header">{logo_html}<div><div class="midia-title">Мидия</div><div class="midia-sub">Многослойный скрининг анемии и дефицитных состояний</div><div class="midia-slogan">Мы не автоматизируем эмпатию — мы освобождаем для неё место</div><div class="midia-hack">Кейс Сеченовского университета · разработано в рамках хакатона · октябрь 2026</div></div><div class="midia-badge">врачебный прототип</div></div>''',
        unsafe_allow_html=True,
    )


def _render_reference_popover() -> None:
    st.markdown("### Справочник")
    rows = []
    for code, label in CLASSES.items():
        rows.append(
            f'<div class="midia-ref-row"><div class="midia-ref-code">{_safe_html(_pretty_code(code))}</div><div class="midia-ref-label">{_safe_html(label)}</div></div>'
        )
    st.markdown('<div class="midia-ref-grid">' + ''.join(rows) + '</div>', unsafe_allow_html=True)
    with st.expander("Источники", expanded=False):
        for source in all_sources():
            st.markdown(f"**{source.get('title', source.get('id'))}**")
            if source.get("note"):
                st.caption(source["note"])


def _render_about_popover() -> None:
    st.markdown("### О проекте")
    st.markdown("**Бренд:** «Мидия». В интерфейсе используется русское название; `MIDIA` — только техническое латинское написание. Смысловая расшифровка: **«Многослойная интеллектуальная диагностическая интерпретация анализов»**.")
    st.caption("Слоган: «Мы не автоматизируем эмпатию — мы освобождаем для неё место».")
    st.markdown("**Как пользоваться:** загрузите или внесите данные → проверьте карточку → выполните скрининг → переключайтесь между слоями результата → сформируйте PDF.")
    st.markdown("**7 слоёв:** нормализация → клиническая логика → ML → карта неопределённости → причинные связи → похожие профили → ИИ-пояснение.")

    st.markdown("**Статус и назначение**")
    st.markdown(
        "- «Мидия» — исследовательский прототип системы поддержки принятия врачебных решений и **не является зарегистрированным медицинским изделием**;\n"
        "- сервис не проходил государственную регистрацию и внешнюю клиническую валидацию и не предназначен для реального клинического применения без дополнительной проверки;\n"
        "- сервис предназначен **только для медицинских работников**; пациентский лист — материал для обсуждения с врачом, а не инструмент самодиагностики;\n"
        "- результат — рабочая скрининговая гипотеза, а не диагноз; лечение и дозировки сервисом не назначаются;\n"
        "- Model score — относительный рейтинг классов и **не является вероятностью диагноза**."
    )

    st.markdown("**Данные и валидация**")
    st.markdown(
        "Модель обучена на предоставленном организатором частично синтетическом датасете: **840 наблюдений, 12 классов**. "
        "Метрики качества являются внутренними и не подтверждены независимой внешней клинической валидацией. "
        "Сервис анализирует один срез лабораторных данных и не учитывает продольную динамику пациента."
    )

    st.markdown("**Область применения**")
    st.markdown(
        "Прототип рассчитан на **совершеннолетних небеременных пациентов в плановом скрининге**. "
        "Беременность, возраст младше 18 лет и неотложные состояния находятся вне области применения. "
        "При подозрении на гемолиз или выраженную почечную дисфункцию требуется дополнительное клиническое подтверждение."
    )

    st.markdown("**Ответственность за входные данные**")
    st.markdown(
        "Врач обязан самостоятельно проверить корректность импортированных лабораторных значений и единиц измерения перед расчётом. "
        "Результат на неполных, ошибочных или некорректно распознанных данных может быть недостоверным."
    )

    st.markdown("**ИИ-пояснение**")
    st.markdown(
        "Локальная языковая модель используется только как вспомогательный слой объяснения и структурирования текста. "
        "Она может допускать неточности формулировок, **не изменяет числовые результаты и class_code** и не является источником клинического решения."
    )

    st.markdown("**Данные и конфиденциальность**")
    st.markdown(
        "ФИО используется только для печатного отчёта и не передаётся в ML-модель или локальную языковую модель. "
        "Жалобы, анамнез и препараты не используются ML-моделью; при включении опционального локального ИИ они могут обрабатываться локально для структурирования клинического контекста. "
        "Конфиденциальные данные не должны передаваться третьим лицам через этот прототип."
    )

    st.markdown("**Похожие профили, МКБ-10 и локальная база**")
    st.markdown(
        "Похожие профили из референс-выборки и локальной базы используются только как справочная информация и не являются самостоятельным клиническим доказательством. "
        "Код МКБ-10 отображается как справочная подсказка; окончательный код выбирает врач после клинической верификации. "
        "Локальная база подтверждённых случаев используется для поиска похожих профилей и **не переобучает ML-модель автоматически**."
    )

    st.markdown(
        "**Перед реальным внедрением необходимы:** внешняя клиническая валидация, информационно-безопасностная и юридическая проверки, "
        "а также валидация на целевой популяции и в инфраструктуре медицинской организации."
    )
    st.markdown(f"**Текущая база:** 840 записей основного кейса · локально подтверждённых случаев: {len(list_cases())}.")

def _render_tools_popover() -> None:
    st.markdown("### Служебные инструменты")
    tab_cases, tab_books = st.tabs(["Подтверждённые случаи", "Локальные PDF для RAG"])
    with tab_cases:
        stored = list_cases()
        st.caption("Здесь можно просматривать и удалять обезличенные подтверждённые случаи. Подтверждение текущего пациента доступно внизу после скрининга.")
        if stored:
            st.dataframe(pd.DataFrame([{
                "ID": x.get("case_id"),
                "Класс": CLASSES.get(x.get("class_code"), x.get("class_code")),
                "Диагноз": x.get("confirmed_diagnosis") or "—",
                "МКБ-10": x.get("icd10") or "—",
            } for x in stored]), hide_index=True, use_container_width=True, height=220)
            delete_id = st.selectbox("Удалить случай", ["—"] + [x.get("case_id") for x in stored], key="tool_case_delete_select")
            if st.button("Удалить выбранный", disabled=delete_id == "—", key="tool_case_delete"):
                if delete_case(delete_id):
                    st.success("Случай удалён.")
                    st.rerun()
        else:
            st.caption("Локальная база пока пуста.")
    with tab_books:
        st.caption("Локальная библиотека разрешённых рекомендаций/протоколов для RAG. Документы пациентов сюда не загружаются.")
        BOOKS_DIR.mkdir(parents=True, exist_ok=True)
        book_upload_revision = int(st.session_state.get("tool_book_upload_revision", 0))
        with st.container(key="tool_book_upload_wrap"):
            book_upload = st.file_uploader(
                "PDF-источник",
                type=["pdf"],
                key=f"tool_book_upload_{book_upload_revision}",
                accept_multiple_files=False,
                label_visibility="collapsed",
            )
            if book_upload is None:
                st.markdown(
                    '''<style>
                    .st-key-tool_book_upload_wrap [data-testid="stFileUploaderDropzone"] button {
                        font-size:0!important;color:transparent!important;background:var(--mid-card)!important;
                        border:1px solid var(--mid-border)!important;box-shadow:none!important;min-width:112px!important;
                        padding:9px 16px!important;border-radius:999px!important;
                    }
                    .st-key-tool_book_upload_wrap [data-testid="stFileUploaderDropzone"] button * {display:none!important;}
                    .st-key-tool_book_upload_wrap [data-testid="stFileUploaderDropzone"] button::after {
                        content:"Загрузка"!important;font-size:14px!important;line-height:1!important;font-weight:600!important;
                        color:var(--mid-ink)!important;display:inline-block!important;
                    }
                    .st-key-tool_book_upload_wrap [data-testid="stFileUploaderDropzone"] button:hover {
                        border-color:var(--mid-blue)!important;background:var(--mid-pale)!important;
                    }
                    </style>''',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '''<style>
                    .st-key-tool_book_upload_wrap [data-testid="stFileUploaderDropzone"],
                    .st-key-tool_book_upload_wrap [data-testid="stFileUploaderFile"] {display:none!important;}
                    .midia-tool-file-card {
                        display:flex;align-items:center;gap:12px;width:100%;box-sizing:border-box;
                        padding:12px 14px;margin:2px 0 10px;background:var(--mid-card);
                        border:1px solid var(--mid-border);border-radius:14px;
                        box-shadow:0 4px 14px rgba(50,63,166,.05);
                    }
                    .midia-tool-file-card__icon {
                        width:34px;height:34px;display:flex;align-items:center;justify-content:center;
                        border-radius:10px;background:var(--mid-pale);color:var(--mid-blue);
                        font-size:17px;flex:0 0 34px;
                    }
                    .midia-tool-file-card__name {color:var(--mid-text);font-size:14px;font-weight:600;line-height:1.25;overflow-wrap:anywhere;}
                    .midia-tool-file-card__meta {color:var(--mid-muted);font-size:12px;line-height:1.25;margin-top:2px;}
                    </style>''',
                    unsafe_allow_html=True,
                )
                file_size = int(getattr(book_upload, "size", 0) or 0)
                if file_size >= 1024 * 1024:
                    size_label = f"{file_size / (1024 * 1024):.1f} МБ"
                elif file_size >= 1024:
                    size_label = f"{file_size / 1024:.1f} КБ"
                else:
                    size_label = f"{file_size} Б" if file_size else "Файл выбран"
                st.markdown(
                    f'<div class="midia-tool-file-card"><div class="midia-tool-file-card__icon">▣</div><div><div class="midia-tool-file-card__name">{html.escape(book_upload.name)}</div><div class="midia-tool-file-card__meta">{html.escape(size_label)}</div></div></div>',
                    unsafe_allow_html=True,
                )
                action_col, delete_col = st.columns([1, 1])
                with action_col:
                    add_source = st.button("Добавить источник", key="tool_book_add", type="primary", use_container_width=True)
                with delete_col:
                    st.button("Удалить", key="tool_book_upload_delete", on_click=_clear_tool_book_upload, type="secondary", use_container_width=True)
                if add_source:
                    try:
                        name = _safe_book_name(book_upload.name)
                        (BOOKS_DIR / name).write_bytes(book_upload.getvalue())
                        refresh_knowledge_cache()
                        st.success(f"Добавлен источник: {name}")
                        _clear_tool_book_upload()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Не удалось добавить PDF: {exc}")
        book_files = sorted(p.name for p in BOOKS_DIR.glob("*.pdf"))
        if book_files:
            to_delete = st.selectbox("Удалить PDF", ["—"] + book_files, key="tool_book_delete_select")
            if st.button("Удалить выбранный PDF", disabled=to_delete == "—", key="tool_book_delete"):
                (BOOKS_DIR / to_delete).unlink(missing_ok=True)
                refresh_knowledge_cache()
                st.success("Источник удалён.")
                st.rerun()
        else:
            st.caption("Локальных PDF пока нет.")


def _render_bars(items: list[tuple[str, float]], maximum: int = 5) -> None:
    for label, value in items[:maximum]:
        pct = max(0.0, min(100.0, float(value)))
        st.markdown(
            f'''<div class="midia-bar-row"><div class="midia-bar-label">{_safe_html(label)}</div><div class="midia-bar-track"><div class="midia-bar-fill" style="width:{pct:.1f}%"></div></div><div class="midia-bar-value">{pct:.1f}</div></div>''',
            unsafe_allow_html=True,
        )


def _render_key_facts(brief: dict) -> None:
    facts = brief.get("anchors_visual") or []
    if not facts:
        return
    blocks = []
    for item in facts[:6]:
        direction = item.get("direction", "•")
        status = item.get("status", "")
        blocks.append(
            f'<div class="midia-fact"><strong>{_safe_html(item.get("label"))}: {_safe_html(f"{item.get("value"):g} {item.get("unit", "")}" if isinstance(item.get("value"), (int,float)) else item.get("value"))}</strong><span class="abn">{_safe_html(direction)} {_safe_html(status)}</span></div>'
        )
    st.markdown('<div class="midia-fact-grid">' + ''.join(blocks) + '</div>', unsafe_allow_html=True)


def _render_causal_chains(result: dict, patient_mode: bool = False) -> None:
    comm = result.get("patient_conversation") or build_patient_conversation(result)
    comm_chains = comm.get("causal_simple", [])
    chains = result.get("causal_chains", [])
    limit = 1 if patient_mode else 3
    for idx, chain in enumerate(chains[:limit]):
        simple = comm_chains[idx] if idx < len(comm_chains) else {}
        markers = chain.get("markers", [])[:3]
        feels = simple.get("patient_feels", [])[:4]
        causes = chain.get("causes", [])[:2]
        observed = '<br>'.join('• ' + _safe_html(x) for x in markers) or '—'
        symptoms = '<br>'.join('• ' + _safe_html(x) for x in feels) or 'не указаны'
        context = '<br>'.join('• ' + _safe_html(x) for x in causes) or 'требует уточнения'
        st.markdown(
            f'''<div class="midia-chain"><div class="midia-chain-card"><b>Что видим в анализах</b>{observed}</div><div class="midia-arrow">→</div><div class="midia-chain-card"><b>Рабочая гипотеза</b>{_safe_html(chain.get("hypothesis", chain.get("label", "")))}</div><div class="midia-arrow">→</div><div class="midia-chain-card"><b>Что чувствует пациент</b>{symptoms}</div><div class="midia-arrow">→</div><div class="midia-chain-card"><b>Что может быть связано</b>{context}</div></div>''',
            unsafe_allow_html=True,
        )


def _translated_exclusions(items: list[dict]) -> pd.DataFrame:
    status = {"abnormal": "есть лабораторный сигнал", "incomplete": "не хватает данных", "no_screening_signal": "сигнал не выявлен"}
    return pd.DataFrame([{
        "Направление": item.get("title", ""),
        "Статус": status.get(item.get("status"), item.get("status", "")),
        "Что видно": item.get("text", "—") or "—",
    } for item in items])


def _icd_catalog() -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for items in ICD10_BY_CLASS.values():
        for item in items:
            pair = (str(item.get("code", "")).strip(), str(item.get("label", "")).strip())
            if pair[0] and pair not in seen:
                seen.add(pair)
                out.append(pair)
    return out


def _render_quick_confirm(result: dict) -> None:
    with st.expander("Подтвердить текущий случай", expanded=False):
        st.caption("После клинического подтверждения можно добавить обезличенный профиль в локальную базу похожих случаев.")
        final = result.get("final", {})
        codes = list(CLASSES)
        default_idx = codes.index(final.get("class_code")) if final.get("class_code") in codes else 0
        code = st.selectbox("Подтверждённый класс", codes, index=default_idx, format_func=lambda x: CLASSES[x], key="quick_case_class")
        diag = st.text_input("Диагноз врача", key="quick_case_diag")

        catalog = _icd_catalog()
        preferred = [(str(x.get("code", "")), str(x.get("label", ""))) for x in ICD10_BY_CLASS.get(code, [])]
        query = diag.strip().lower().replace("ё", "е")
        def _icd_rank(item: tuple[str, str]) -> tuple[int, int]:
            item_code, item_label = item
            code_low = item_code.lower()
            label_low = item_label.lower().replace("ё", "е")
            score = 0
            if query:
                if code_low.startswith(query):
                    score += 100
                elif query in code_low:
                    score += 80
                if query in label_low:
                    score += 70
                for token in [x for x in re.split(r"[^a-zа-я0-9]+", query) if len(x) >= 3]:
                    if token in label_low:
                        score += 12
            if item in preferred:
                score += 25
            return (score, -catalog.index(item))
        ordered = sorted(catalog, key=_icd_rank, reverse=True)
        if query and ordered and _icd_rank(ordered[0])[0] > 0:
            st.caption(f"Подходящий вариант: {ordered[0][0]} — {ordered[0][1]}")
        options: list[tuple[str, str] | None] = [None] + ordered
        selected_icd = st.selectbox(
            "МКБ-10",
            options,
            format_func=lambda x: "Не выбран" if x is None else f"{x[0]} — {x[1]}",
            key="quick_case_icd_select",
            help="Начните вводить код или название в раскрытом списке — варианты будут отфильтрованы. Полное название показывается рядом с кодом.",
        )
        icd = selected_icd[0] if selected_icd else ""
        if st.button("Добавить подтверждённый случай", key="quick_case_add", type="primary"):
            try:
                added = add_case(result["patient"], code, diag, icd)
                st.success(f"Случай {added['case_id']} сохранён обезличенно.")
            except Exception as exc:
                st.error(f"Не удалось сохранить случай: {exc}")


# --- App -----------------------------------------------------------------
# Порядок экрана: шапка и инструменты -> подтверждение права обработки данных ->
# карточка пациента -> импорт файла -> проверка значений и единиц -> скрининг ->
# слои результата -> отчёты (врач / пациент).
_ensure_state()
_inject_css(bool(st.session_state.get("dark_theme", False)))
_render_header()

if st.session_state.get("pending_import_record") is not None:
    _apply_import(st.session_state.pop("pending_import_record"))

current_result = st.session_state.get("result")
ref_col, about_col, tools_col, theme_col, new_col = st.columns([1.0, 1.0, 1.0, 1.0, 1.35], vertical_alignment="center")
with ref_col:
    with st.popover("Справочник", use_container_width=True):
        _render_reference_popover()
with about_col:
    with st.popover("О проекте", use_container_width=True):
        _render_about_popover()
with tools_col:
    with st.popover("Инструменты", use_container_width=True):
        _render_tools_popover()
with theme_col:
    st.toggle("Тёмная тема", key="dark_theme")
with new_col:
    st.button("＋ Новый пациент", on_click=_reset, use_container_width=True, type="primary")

st.markdown(f'<div class="midia-disclaimer">{_safe_html(UI_DISCLAIMER)}</div>', unsafe_allow_html=True)

authorized = st.checkbox(
    "Подтверждаю, что являюсь медицинским работником / уполномоченным пользователем, имею законное основание для обработки внесённых данных пациента и ознакомлен(а) с дисклеймером и информацией в разделе «О проекте».",
    key="authorization",
    help="Это подтверждение не заменяет локальные документы медицинской организации и отдельную юридическую проверку процесса.",
)

st.subheader("Карточка пациента")
left_top, right_top = st.columns(2, vertical_alignment="bottom")
with left_top:
    st.text_input(
        "ФИО пациента",
        key="full_name",
        help="ФИО используется только для печати отчёта и не сохраняется в базе. Конфиденциальные данные не передаются третьим лицам.",
    )
with right_top:
    age_col, sex_col = st.columns([0.8, 1.2], vertical_alignment="bottom")
    age_col.text_input("Возраст, лет", key="age", help="Проект предназначен для совершеннолетних пациентов.")
    sex_col.selectbox("Пол", ["Не указан", "Женский", "Мужской"], key="sex", help="Пол необходимо указать для корректного пола-специфического порога гемоглобина.")

left, right = st.columns(2)
left.text_area("Жалобы / симптомы", key="complaints", height=105)
right.text_area("Анамнез / хронические заболевания", key="history_text", height=105)
st.text_area("Постоянные препараты", key="medications", height=75)

st.subheader("Импорт карточки пациента и лабораторных показателей")
if st.session_state.get("input_mode") == "Импортировать файл":
    st.session_state["input_mode"] = "Загрузить файл"
input_mode = st.radio("Способ ввода", ["Загрузить файл", "Ввести вручную"], horizontal=True, key="input_mode", label_visibility="collapsed")

if input_mode == "Загрузить файл":
    st.markdown(
        '<div class="midia-import-note"><strong>Форматы импорта:</strong> PDF, TXT, CSV, TSV, JSON, XLSX, XLS, HTML</div>',
        unsafe_allow_html=True,
    )
    upload_revision = int(st.session_state.get("patient_upload_revision", 0))
    with st.container(key="patient_upload_wrap"):
        upload = st.file_uploader(
            "Файл пациента",
            type=None,
            key=f"patient_upload_{upload_revision}",
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        if upload is None:
            st.markdown(
                '''<style>
                .st-key-patient_upload_wrap [data-testid="stFileUploaderDropzone"] button {
                    font-size:0!important;color:transparent!important;background:var(--mid-card)!important;
                    border:1px solid var(--mid-border)!important;box-shadow:none!important;min-width:112px!important;
                    padding:9px 16px!important;border-radius:999px!important;
                }
                .st-key-patient_upload_wrap [data-testid="stFileUploaderDropzone"] button * {display:none!important;}
                .st-key-patient_upload_wrap [data-testid="stFileUploaderDropzone"] button::after {
                    content:"Загрузка"!important;font-size:14px!important;line-height:1!important;font-weight:600!important;
                    color:var(--mid-ink)!important;display:inline-block!important;
                }
                .st-key-patient_upload_wrap [data-testid="stFileUploaderDropzone"] button:hover {
                    border-color:var(--mid-blue)!important;background:var(--mid-pale)!important;
                }
                </style>''',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '''<style>
                /* После выбора скрываем нативные элементы повторной загрузки.
                   Информация о выбранном файле выводится отдельной стабильной карточкой ниже. */
                .st-key-patient_upload_wrap [data-testid="stFileUploaderDropzone"],
                .st-key-patient_upload_wrap [data-testid="stFileUploaderFile"] {
                    display:none!important;
                }
                .midia-file-card {
                    display:flex;align-items:center;gap:12px;
                    width:100%;box-sizing:border-box;
                    padding:12px 14px;margin:2px 0 10px;
                    background:var(--mid-card);border:1px solid var(--mid-border);
                    border-radius:14px;box-shadow:0 4px 14px rgba(50,63,166,.05);
                }
                .midia-file-card__icon {
                    width:34px;height:34px;display:flex;align-items:center;justify-content:center;
                    border-radius:10px;background:var(--mid-pale);color:var(--mid-blue);
                    font-size:17px;flex:0 0 34px;
                }
                .midia-file-card__name {
                    color:var(--mid-text);font-size:14px;font-weight:600;line-height:1.25;overflow-wrap:anywhere;
                }
                .midia-file-card__meta {
                    color:var(--mid-muted);font-size:12px;line-height:1.25;margin-top:2px;
                }
                </style>''',
                unsafe_allow_html=True,
            )
            file_size = int(getattr(upload, "size", 0) or 0)
            if file_size >= 1024 * 1024:
                size_label = f"{file_size / (1024 * 1024):.1f} МБ"
            elif file_size >= 1024:
                size_label = f"{file_size / 1024:.1f} КБ"
            else:
                size_label = f"{file_size} Б" if file_size else "Файл выбран"
            st.markdown(
                f'<div class="midia-file-card"><div class="midia-file-card__icon">▣</div><div><div class="midia-file-card__name">{html.escape(upload.name)}</div><div class="midia-file-card__meta">{html.escape(size_label)}</div></div></div>',
                unsafe_allow_html=True,
            )
            st.button("Удалить", key="patient_upload_delete", on_click=_clear_patient_upload, type="secondary")
    st.checkbox(
        "Помочь разобрать клинический контекст локальной нейросети",
        key="use_ai_import",
        help="Опционально: локальная нейросеть помогает структурировать менее явный клинический текст в жалобы, анамнез и препараты. ФИО перед ней скрывается; лабораторные значения, диагноз и ML-расчёт она не меняет. При недоступной нейросети используется обычный парсер.",
    )
    if st.button("Распознать файл", disabled=upload is None, type="primary"):
        try:
            data = upload.getvalue()
            records, messages = read_upload(data, upload.name)
            record = records[0]
            if st.session_state.get("use_ai_import"):
                with st.spinner("Локальная нейросеть помогает структурировать клинический контекст…"):
                    ai_ctx = structure_clinical_context(data, upload.name)
                if ai_ctx.get("ok"):
                    for src, dst in (("complaints", "complaints"), ("history", "history"), ("medications", "medications")):
                        value = str(ai_ctx.get("data", {}).get(src, "") or "").strip()
                        if value:
                            record[dst] = value
                    st.session_state["ai_import_message"] = "Клинический контекст дополнительно структурирован локальной нейросетью."
                elif ai_ctx.get("warning"):
                    st.session_state["ai_import_message"] = ai_ctx["warning"]
            st.session_state["pending_import_record"] = record
            st.session_state["import_messages"] = messages
            st.session_state["import_ok"] = True
            st.rerun()
        except Exception as exc:
            st.session_state["import_ok"] = False
            st.error(str(exc))
    if st.session_state.get("import_ok"):
        st.success("Файл распознан. Доступные поля карточки и лабораторные показатели предзаполнены.")
        conflicts = [m for m in st.session_state.get("import_messages", []) if "дублик" in m.lower() or "разными знач" in m.lower()]
        if conflicts:
            st.warning("⚠️ " + " ".join(conflicts) + " Любое поле можно исправить вручную перед расчётом.")
        else:
            st.warning("Перед расчётом сверьте распознанные значения и единицы с исходным документом. Любое поле можно исправить вручную.")
        if st.session_state.get("ai_import_message"):
            st.caption(st.session_state.get("ai_import_message"))

st.markdown("**Проверка и редактирование перед расчётом**")
st.caption(UNIT_NOTE)
unit_options = sorted({str(v.get("unit", "")) for v in LABS.values() if v.get("unit")} | {"г/дл", "мг/дл", "нг/мл", "пмоль/л", "нмоль/л"})
editor_frame = pd.DataFrame(st.session_state["rows"])
editor_frame["Удалить"] = False
rows = st.data_editor(
    editor_frame,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key="editor_" + str(st.session_state.get("revision", 0)),
    column_order=["Показатель", "Значение", "Единица", "Удалить"],
    column_config={
        "Показатель": st.column_config.SelectboxColumn(options=[v["label"] for v in LABS.values()], required=True, width="large"),
        "Значение": st.column_config.TextColumn(width="medium"),
        "Единица": st.column_config.SelectboxColumn(options=unit_options, width="medium"),
        "Удалить": st.column_config.CheckboxColumn(help="Отметьте ошибочно добавленную или дублирующую строку", width="small"),
    },
)
delete_mask = rows.get("Удалить", pd.Series(False, index=rows.index)).fillna(False).astype(bool)
if st.button("Удалить выбранные строки", disabled=not bool(delete_mask.any()), type="secondary"):
    kept = rows.loc[~delete_mask, ["Показатель", "Значение", "Единица"]].fillna("")
    st.session_state["rows"] = kept.to_dict("records") or _default_rows()
    st.session_state["revision"] = st.session_state.get("revision", 0) + 1
    st.rerun()
rows = rows.drop(columns=["Удалить"], errors="ignore")
rows, units_changed = _suggest_units(rows)
if units_changed:
    st.session_state["rows"] = rows.fillna("").to_dict("records")
    st.session_state["revision"] = st.session_state.get("revision", 0) + 1
    st.rerun()

units_confirmed = st.checkbox("Если единица в исходном бланке отсутствует, считать указанную в таблице единицу подтверждённой", value=False)
raw, form_errors = _raw_from_form(rows, units_confirmed)
for error in form_errors:
    st.error(error)
if not authorized:
    st.info("Перед расчётом подтвердите статус уполномоченного пользователя и законное основание для обработки данных.")

# Скрининг доступен только при корректной форме и подтверждённом праве обработки данных.
# ФИО в screen() не передаётся: оно нужно только для печатной формы PDF.
if st.button("Выполнить скрининг", type="primary", disabled=bool(form_errors) or not authorized):
    try:
        with st.spinner("Выполняется скрининг…"):
            result = screen(raw, units_confirmed=units_confirmed)
        st.session_state["result"] = result
        st.session_state["result_fingerprint"] = _fingerprint(raw, units_confirmed)
        st.session_state.pop("ai_explanation", None)
    except Exception as exc:
        st.error(str(exc))

result = st.session_state.get("result")
if result:
    stale = st.session_state.get("result_fingerprint") != _fingerprint(raw, units_confirmed)
    if stale:
        st.warning("Карточка или анализы изменены после расчёта. Пересчитайте результат перед выгрузкой отчёта.")

    final = result["final"]
    clinical = result["clinical"]
    result["patient_conversation"] = build_patient_conversation(result)
    brief = result.get("doctor_brief", {})
    # Rebuild brief from communication-compatible information when older result lacks visual anchors.
    if not brief.get("anchors_visual"):
        from screening.communication import build_doctor_brief
        brief = build_doctor_brief(result)
        result["doctor_brief"] = brief

    st.divider()
    st.subheader("Предварительный результат")
    st.markdown(f'<div class="midia-summary"><h3>{_safe_html(final.get("label", ""))}</h3><div class="midia-code">Класс: {_safe_html(_pretty_code(final.get("class_code")))} · {_safe_html(final.get("confidence_band", ""))}</div></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Ключевое для врача")
        if final.get("label"):
            st.markdown(f"**Ведущая скрининговая гипотеза:** {final['label']}")
        best = result.get("uncertainty", {}).get("best_next_measurement")
        if best:
            st.markdown(f"**Наиболее информативное недостающее измерение:** {best.get('label')}.")
        _render_key_facts(brief)
        st.markdown('<div class="midia-doctor-inner-gap"></div>', unsafe_allow_html=True)

    st.markdown('<div class="midia-doctor-section-gap"></div>', unsafe_allow_html=True)
    conversation_mode = st.toggle(
        "Режим разговора с пациентом",
        value=False,
        key="conversation_mode",
        help="Наглядное представление результата для совместного просмотра с пациентом: только ключевые факты, рабочая гипотеза, неопределённость и следующий шаг.",
    )

    for alert in clinical.get("alerts", []):
        st.error(alert)

    if conversation_mode:
        comm = result.get("patient_conversation", {})
        with st.container(border=True):
            st.markdown("### Наглядное объяснение результата")
            st.markdown("**Что мы уже видим**")
            st.write(comm.get("plain_summary", ""))
            ranking = comm.get("model_ranking", [])
            if ranking:
                st.markdown("**Как система ранжирует рабочие варианты**")
                _render_bars([(x.get("label", ""), x.get("rating", 0)) for x in ranking], maximum=3)
            st.markdown("**Как могут быть связаны данные**")
            _render_causal_chains(result, patient_mode=True)
            st.markdown("**Что пока остаётся неопределённым**")
            st.write(comm.get("uncertainty_plain", ""))
            st.markdown("**Что поможет уточнить картину**")
            st.info(comm.get("why_next_step", ""))
    else:
        t0, t1, t2, t3, t4, t5, t6 = st.tabs([
            "Нормализация", "Клиническая логика", "ML", "Неопределённость", "Причины", "Похожие профили", "ИИ"
        ])

        with t0:
            a, b, c = st.columns(3)
            a.metric("Показателей в расчёте", len(result.get("patient", {}).get("labs", {})))
            b.metric("Возраст", f"{result['patient'].get('age_years', '—'):g}" if isinstance(result['patient'].get('age_years'), (int,float)) else result['patient'].get('age_years', '—'))
            bsex = "Женский" if result['patient'].get('sex') == 'F' else "Мужской" if result['patient'].get('sex') == 'M' else "—"
            c.metric("Пол", bsex)
            if result.get("warnings"):
                for warning in result.get("warnings", []):
                    st.warning(warning)
            else:
                st.success("Данные подготовлены к расчёту без технических предупреждений.")

        with t1:
            hb = clinical.get("hemoglobin")
            if hb is not None:
                sex_text = "женского пола" if result["patient"].get("sex") == "F" else "мужского пола"
                st.markdown(f"**Гемоглобин {hb:g} г/л** · порог анемии для пациента {sex_text}: **{clinical.get('threshold'):g} г/л**")
            for finding in clinical.get("findings", []):
                if finding.get("code") == "hb_threshold":
                    continue
                finding_class = "clinical-finding warning" if finding.get("severity") == "warning" else "clinical-finding"
                st.markdown(
                    f'<div class="{finding_class}">• {_safe_html(finding.get("text", ""))}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("#### Проверка других причин анемии")
            exclusions = clinical.get("exclusions", [])
            checked = [x for x in exclusions if x.get("status") != "incomplete"]
            incomplete = [x for x in exclusions if x.get("status") == "incomplete"]
            st.caption(f"Проверяются дополнительные направления: {', '.join(x.get('title','') for x in exclusions)}.")
            if checked:
                exclusion_df = _translated_exclusions(checked)
                exclusion_style = exclusion_df.style.map(
                    lambda value: (
                        "color:#2E7D57;font-weight:700" if str(value) == "сигнал не выявлен"
                        else "color:#C9434C;font-weight:700" if str(value) == "есть лабораторный сигнал"
                        else ""
                    ),
                    subset=["Статус"],
                )
                st.dataframe(exclusion_style, hide_index=True, use_container_width=True)
            if incomplete:
                with st.expander("Направления, для которых не хватает данных", expanded=False):
                    for item in incomplete:
                        st.write(f"**{item.get('title')}** — не хватает: {', '.join(item.get('missing', [])) or 'данных'}")

            st.markdown("#### Предположительный код по МКБ-10")
            st.caption("Код используется как справочная подсказка и не является финальным диагнозом.")
            coding = result.get("coding", {})
            if coding.get("candidates"):
                rows_icd = [{"Код": x.get("code"), "Название": x.get("label"), "Когда используется": x.get("when")} for x in coding.get("candidates", [])]
                st.dataframe(pd.DataFrame(rows_icd), hide_index=True, use_container_width=True)
            else:
                st.write("Для текущего класса кандидат кода не предложен.")

        with t2:
            model = result.get("model")
            if model:
                ranked = list(model.get("class_scores", {}).items())
                st.markdown("#### Рейтинг рабочих классов")
                _render_bars([(CLASSES.get(code, code), float(score) * 100) for code, score in ranked], maximum=5)
                if len(ranked) >= 2:
                    gap = (float(ranked[0][1]) - float(ranked[1][1])) * 100
                    st.caption(f"Отрыв первого класса от второго: {gap:.1f} пункта модельного рейтинга.")
                sensitivity = model.get("sensitivity") or []
                if sensitivity:
                    st.markdown("#### Какие показатели сильнее меняют модельный рейтинг")
                    max_abs = max(abs(float(x.get("score_change", 0))) for x in sensitivity) or 1.0
                    items = []
                    for item in sensitivity[:5]:
                        label = LABS.get(item.get("feature"), {}).get("label", item.get("feature", ""))
                        strength = abs(float(item.get("score_change", 0))) / max_abs * 100
                        items.append((label, strength))
                    _render_bars(items, maximum=5)
            else:
                st.info("ML-слой недоступен.")

        with t3:
            unc = result.get("uncertainty", {})
            if unc.get("top_hypothesis"):
                top = unc.get("top_hypothesis") or {}
                alt = unc.get("alternative") or {}
                top_rating = float(top.get("score", 0)) * 100
                alt_rating = float(alt.get("score", 0)) * 100
                st.markdown(f"### {top.get('label','')} ↔ {alt.get('label','')}")
                st.markdown(f"**Модельный рейтинг:** {top_rating:.1f}/100 ↔ {alt_rating:.1f}/100 · **отрыв {abs(top_rating-alt_rating):.1f} пункта**")
                if unc.get("ambiguity_factors"):
                    st.markdown("**Почему остаётся неопределённость**")
                    for factor in unc.get("ambiguity_factors", [])[:4]:
                        st.write(f"• **{factor.get('factor')}** — {factor.get('text')}")
                best = unc.get("best_next_measurement")
                if best:
                    st.success(f"**{best.get('label')}** поможет лучше различить две ведущие гипотезы и может уменьшить неопределённость диагноза.")
                items = unc.get("items") or []
                if items:
                    max_score = max(float(x.get("score", 0)) for x in items) or 1.0
                    rows_unc = []
                    for item in items[:5]:
                        rows_unc.append({
                            "Показатель": item.get("label", ""),
                            "Приоритет уточнения": f"{float(item.get('score',0))/max_score*100:.1f}/100",
                            "Почему полезен": str(item.get("why", "")).replace("предоставленном датасете", "предоставленных данных"),
                        })
                    st.dataframe(pd.DataFrame(rows_unc), hide_index=True, use_container_width=True)
            else:
                st.info(unc.get("message", "Карта неопределённости недоступна."))
            coexist = result.get("coexisting_signals", [])
            if coexist:
                st.markdown("**Сопутствующие сигналы**")
                for item in coexist:
                    st.write("•", item.get("label"))

        with t4:
            _render_causal_chains(result, patient_mode=False)

        with t5:
            local_refs = local_confirmed_neighbors(result["patient"], k=5)
            result.setdefault("similar_cases", {})["local_confirmed"] = local_refs
            st.markdown("#### Локально подтверждённые похожие случаи")
            if local_refs:
                st.dataframe(pd.DataFrame([{
                    "Сходство": f"{item.get('similarity',0)*100:.0f}%",
                    "Подтверждённый класс": CLASSES.get(item.get("class_code"), item.get("class_code")),
                    "Диагноз врача": item.get("confirmed_diagnosis") or "—",
                    "Возраст": item.get("age_years"),
                    "Пол": "Ж" if item.get("sex") == "F" else "М" if item.get("sex") == "M" else item.get("sex"),
                } for item in local_refs]), hide_index=True, use_container_width=True)
            else:
                st.info("Локально подтверждённых похожих случаев пока нет.")
            st.markdown("#### Похожие профили из референс-выборки")
            refs = result.get("similar_cases", {}).get("reference_dataset", [])
            if refs:
                st.dataframe(pd.DataFrame([{
                    "Сходство": f"{item.get('similarity',0)*100:.0f}%",
                    "Класс": CLASSES.get(item.get("class_code"), item.get("class_code")),
                    "Причина в выборке": str(item.get("deficiency_cause") or "—").replace("_", " "),
                    "Возраст": item.get("age_years"),
                    "Пол": "Ж" if item.get("sex") == "F" else "М" if item.get("sex") == "M" else item.get("sex"),
                    "Совпало лабораторных признаков": item.get("lab_overlap"),
                } for item in refs[:5]]), hide_index=True, use_container_width=True)
            else:
                st.caption("Похожие профили не найдены.")

        ai = st.session_state.get("ai_explanation")
        with t6:
            status = ollama_status()
            if status["server"] and status["model_installed"]:
                st.success(f"Локальная нейросеть готова · {status['model']}")
            elif status["server"]:
                st.warning(f"Ollama запущена, но модель {status['model']} не найдена.")
            else:
                st.info("Локальная нейросеть недоступна. Остальные слои работают независимо.")
            if st.button("Сформировать / обновить ИИ-пояснение", key="ai_button"):
                progress_bar = st.progress(0, text="Подготовка…")
                def _progress(percent: int, message: str) -> None:
                    progress_bar.progress(percent, text=message)
                st.session_state["ai_explanation"] = generate_ai_explanation(result, result["patient"], progress=_progress)
                ai = st.session_state.get("ai_explanation")
            ai = st.session_state.get("ai_explanation")
            if ai:
                if ai.get("warning"):
                    st.info(ai.get("warning"))
                st.markdown("**Объяснение для врача**")
                st.markdown(ai.get("text", ""))
                if ai.get("patient_text"):
                    st.caption("Короткая пациентская версия сформирована отдельно и будет помещена на последнюю страницу PDF «Лист пациента».")
                if ai.get("sources"):
                    with st.expander("Использованные фрагменты источников", expanded=False):
                        for source in ai.get("sources", []):
                            st.caption(f"{source.get('source')} · релевантность {source.get('score',0):.3f}")
                            st.write(source.get("text", ""))
            else:
                st.caption("ИИ-пояснение ещё не сформировано.")

    ai = st.session_state.get("ai_explanation")
    st.subheader("Отчёты")
    if not stale:
        try:
            doctor_pdf = generate_pdf(
                result,
                patient_name=st.session_state.get("full_name", "").strip(),
                ai_text=ai.get("text") if ai else None,
                ai_patient_text=None,
                include_patient_page=False,
            )
            patient_pdf = generate_patient_pdf(
                result,
                patient_name=st.session_state.get("full_name", "").strip(),
                ai_patient_text=ai.get("patient_text") if ai and ai.get("mode") == "ollama" else None,
                ai_generated=bool(ai and ai.get("mode") == "ollama" and ai.get("patient_text")),
            )
            c1, c2 = st.columns(2)
            c1.download_button("Скачать врачебный PDF", doctor_pdf, "midia_doctor_report.pdf", "application/pdf", use_container_width=True)
            c2.download_button("Скачать PDF пациента", patient_pdf, "midia_patient_sheet.pdf", "application/pdf", use_container_width=True)
        except Exception as exc:
            st.error(f"Не удалось сформировать PDF: {exc}")

    _render_quick_confirm(result)
