"""Генерация PDF: врачебный отчёт и лист пациента (ReportLab)."""

from __future__ import annotations

import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .schema import CLASSES, LABS
from .communication import build_patient_conversation

_FONT_REG = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]
_FONT_BOLD = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]


ROOT = Path(__file__).resolve().parent.parent
ASSET_MIDIA_LOGO = ROOT / "assets" / "midia_logo.png"
ASSET_LOGO_FALLBACK = ROOT / "assets" / "sechenov_logo.png"


def _report_logo(width_mm: float = 34) -> Image | None:
    for asset_path in (ASSET_MIDIA_LOGO, ASSET_LOGO_FALLBACK):
        if not asset_path.exists():
            continue
        try:
            img_w, img_h = ImageReader(str(asset_path)).getSize()
            width = width_mm * mm
            height = width * (float(img_h) / float(img_w)) if img_w else width
            logo = Image(str(asset_path), width=width, height=height)
            logo.hAlign = "LEFT"
            return logo
        except Exception:
            continue
    return None


def _register_fonts() -> tuple[str, str]:
    if "HSBody" in pdfmetrics.getRegisteredFontNames():
        return "HSBody", "HSBold"
    reg = next((p for p in _FONT_REG if os.path.exists(p)), None)
    bold = next((p for p in _FONT_BOLD if os.path.exists(p)), None)
    if reg:
        pdfmetrics.registerFont(TTFont("HSBody", reg))
        pdfmetrics.registerFont(TTFont("HSBold", bold or reg))
        return "HSBody", "HSBold"
    return "Helvetica", "Helvetica-Bold"


def _p(text: Any, style) -> Paragraph:
    return Paragraph(escape(str(text or "")), style)


def _inline_markdown(text: str) -> str:
    """Very small, safe Markdown subset for ReportLab Paragraph.

    LLM output is escaped first; only **bold** and `code` are reintroduced as
    ReportLab markup. Markdown heading markers are handled line-by-line elsewhere.
    """
    value = escape(str(text or ""))
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", value)
    return value


def _markdown_flowables(text: str, normal, small, h2, h3) -> list[Any]:
    """Render common LLM Markdown cleanly instead of printing raw #/* characters."""
    clean_text = re.sub(r"(?i)<<<\s*(?:DOCTOR|PATIENT|ВРАЧ|ПАЦИЕНТ)\s*>>>", "", str(text or ""))
    out: list[Any] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            joined = " ".join(x.strip() for x in paragraph if x.strip())
            if joined:
                out.append(Paragraph(_inline_markdown(joined), normal))
            paragraph.clear()

    for raw in clean_text.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line:
            flush()
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            flush()
            out.append(Paragraph(_inline_markdown(heading.group(2)), h2 if len(heading.group(1)) <= 2 else h3))
            continue
        bullet = re.match(r"^[-*•]\s+(.+)$", line)
        if bullet:
            flush()
            out.append(Paragraph("• " + _inline_markdown(bullet.group(1)), normal))
            continue
        numbered = re.match(r"^(\d+)[.)]\s+(.+)$", line)
        if numbered:
            flush()
            out.append(Paragraph(f"{numbered.group(1)}. " + _inline_markdown(numbered.group(2)), normal))
            continue
        paragraph.append(line)
    flush()
    if not out:
        out.append(_p(clean_text, small))
    return out


def _table(rows: list[list[Any]], widths, style, header: bool = True) -> Table:
    cell_style = ParagraphStyle(
        f"{style.name}Cell",
        parent=style,
        leading=max(float(style.fontSize) + 1.0, float(style.leading or style.fontSize)),
        spaceBefore=0,
        spaceAfter=0,
        alignment=TA_LEFT,
    )
    cooked = [[_p(v, cell_style) for v in row] for row in rows]
    t = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8DCEF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        cmds.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF0FB")))
    t.setStyle(TableStyle(cmds))
    t.spaceBefore = 8
    t.spaceAfter = 13
    return t


def _important_lab_keys(result: dict[str, Any]) -> set[str]:
    important: set[str] = {"hemoglobin"}
    for chain in result.get("causal_chains", []):
        for marker in chain.get("markers", []):
            marker_text = str(marker)
            for key, meta in LABS.items():
                if marker_text.startswith(str(meta.get("label", key)) + " ="):
                    important.add(key)
    return important


def _compact_lab_table(patient: dict[str, Any], result: dict[str, Any], small, bold_style) -> Table:
    important = _important_lab_keys(result)
    items = list(patient.get("labs", {}).items())
    items = sorted(enumerate(items), key=lambda x: (x[1][0] not in important, x[0]))
    pairs = [item for _, item in items]
    rows: list[list[Any]] = []
    highlight_cells: list[tuple[int, int]] = []
    for i in range(0, len(pairs), 2):
        row: list[Any] = []
        for pair_idx in range(2):
            pos = i + pair_idx
            if pos >= len(pairs):
                row.extend([_p("", small), _p("", small)])
                continue
            key, value = pairs[pos]
            label_style = bold_style if key in important else small
            label = LABS[key]["label"]
            unit = patient.get("units", {}).get(key, LABS[key].get("unit", ""))
            row.extend([_p(label, label_style), _p(f"{value:g} {unit}".strip(), label_style if key in important else small)])
            if key in important:
                col = pair_idx * 2
                highlight_cells.extend([(len(rows), col), (len(rows), col + 1)])
        rows.append(row)
    table = Table(rows, colWidths=[55 * mm, 32 * mm, 55 * mm, 32 * mm], hAlign="LEFT")
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D8DCEF")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for r, c in highlight_cells:
        cmds.append(("BACKGROUND", (c, r), (c, r), colors.HexColor("#EEF0FB")))
    table.setStyle(TableStyle(cmds))
    table.spaceBefore = 8
    table.spaceAfter = 13
    return table


def _report_styles():
    body_font, bold_font = _register_fonts()
    styles = getSampleStyleSheet()
    ink = colors.HexColor("#323FA6")
    blue = colors.HexColor("#323FA6")
    pale = colors.HexColor("#EEF0FB")
    line = colors.HexColor("#D8DCEF")
    muted = colors.HexColor("#5D6274")
    # Typography rule for the final reports: only headings are larger.
    # Body text, tables, disclaimers, notes and bordered conclusions all use 9 pt.
    normal = ParagraphStyle("MIDNormal", parent=styles["Normal"], fontName=body_font, fontSize=9, leading=10.8, spaceAfter=7, textColor=colors.HexColor("#222222"), alignment=TA_LEFT)
    small = ParagraphStyle("MIDSmall", parent=normal, fontSize=9, leading=10.8, spaceAfter=5, textColor=colors.HexColor("#222222"), alignment=TA_LEFT)
    small_bold = ParagraphStyle("MIDSmallBold", parent=small, fontName=bold_font, textColor=colors.HexColor("#222222"), alignment=TA_LEFT)
    h1 = ParagraphStyle("MIDTitle", parent=styles["Title"], fontName=bold_font, fontSize=18, leading=21, spaceAfter=10, textColor=ink)
    h2 = ParagraphStyle("MIDH2", parent=styles["Heading2"], fontName=bold_font, fontSize=12, leading=14, spaceBefore=15, spaceAfter=9, keepWithNext=True, textColor=blue, alignment=TA_LEFT)
    h3 = ParagraphStyle("MIDH3", parent=h2, fontSize=10, leading=11.5, spaceBefore=11, spaceAfter=7, textColor=ink, alignment=TA_LEFT)
    # Blue bordered conclusions are deliberately slightly narrower than the text column
    # so their rounded borders never protrude past adjacent text/tables.
    box_indent = 1.5 * mm
    callout = ParagraphStyle("MIDCallout", parent=normal, backColor=pale, borderColor=colors.HexColor("#7A85DE"), borderWidth=0.6, borderPadding=8, borderRadius=10, textColor=colors.HexColor("#222222"), fontSize=9, leading=10.8, spaceBefore=12, spaceAfter=16, alignment=TA_LEFT, leftIndent=box_indent, rightIndent=box_indent)
    warn = ParagraphStyle("MIDWarn", parent=normal, backColor=colors.HexColor("#FFF1F3"), borderColor=colors.HexColor("#E85A62"), borderWidth=0.6, borderPadding=8, borderRadius=10, fontSize=9, leading=10.8, spaceBefore=8, spaceAfter=13, alignment=TA_LEFT, leftIndent=box_indent, rightIndent=box_indent, firstLineIndent=0)
    patient_style = ParagraphStyle("MIDPatient", parent=normal, backColor=colors.HexColor("#F5F6FD"), borderColor=colors.HexColor("#323FA6"), borderWidth=0.6, borderPadding=8, borderRadius=10, fontSize=9, leading=10.8, spaceBefore=12, spaceAfter=16, alignment=TA_LEFT, leftIndent=box_indent, rightIndent=box_indent)
    return normal, small, small_bold, h1, h2, h3, callout, warn, patient_style, ink, blue, pale, line


def _ui_disclaimer() -> str:
    return (
        "Мидия — прототип врачебной системы скрининга. Результат является рабочей гипотезой и не заменяет диагноз, "
        "клиническое решение или локальные документы медицинской организации. Сервис не назначает лечение. "
        "Model score используется только для сравнения классов и не является вероятностью диагноза."
    )


def _pretty_code(code: str | None) -> str:
    if not code:
        return "—"
    return str(code).replace("_", " ")


def _pretty_value(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return str(value).replace("_", " ")


def _patient_indicator_rows(communication: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = [["Показатель", "Значение", "Что это значит"]]
    for item in communication.get("key_facts_visual", [])[:5]:
        rows.append([
            item.get("label", ""),
            f"{item.get('value', ''):g} {item.get('unit', '')}".strip() if isinstance(item.get("value"), (int, float)) else str(item.get("value", "")),
            f"{item.get('direction', '•')} {item.get('status', '')}".strip(),
        ])
    return rows


def generate_pdf(
    result: dict[str, Any],
    patient_name: str = "",
    ai_text: str | None = None,
    ai_patient_text: str | None = None,
    include_patient_page: bool = True,
) -> bytes:
    normal, small, small_bold, h1, h2, h3, callout, warn, patient_style, ink, blue, pale, line = _report_styles()

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Мидия — врачебный отчёт",
    )
    story: list[Any] = []

    patient = result["patient"]
    clinical = result["clinical"]
    final = result["final"]
    model = result.get("model") or {}
    uncertainty = result.get("uncertainty") or {}
    brief = result.get("doctor_brief") or {}
    communication = result.get("patient_conversation") or build_patient_conversation(result)
    coding = result.get("coding") or {}
    ctx = patient.get("context", {})

    logo = _report_logo(18)
    if logo:
        story.append(logo)
        story.append(Spacer(1, 3))

    story.append(_p("Мидия", h1))
    story.append(_p("Многослойный скрининг анемии и дефицитных состояний", h2))
    story.append(_p("Кейс Сеченовского университета · хакатон · октябрь 2026", small))
    story.append(Spacer(1, 4))
    story.append(_p(_ui_disclaimer(), warn))

    story.append(_p("Карточка пациента", h2))
    story.append(_table([
        ["ФИО", patient_name.strip() or "не указано", "Возраст", f"{patient['age_years']:g}"],
        ["Пол", "Женский" if patient["sex"] == "F" else "Мужской", "Дата расчёта", result.get("created_at", "")],
    ], [25 * mm, 60 * mm, 25 * mm, 60 * mm], small, header=False))
    story.append(_table([
        ["Жалобы / симптомы", ctx.get("complaints") or "—"],
        ["Анамнез / хронические заболевания", ctx.get("history") or "—"],
        ["Постоянные препараты", ctx.get("medications") or "—"],
    ], [48 * mm, 127 * mm], small, header=False))

    story.append(_p("Лабораторные показатели", h2))
    story.append(_compact_lab_table(patient, result, small, small_bold))

    story.append(_p("Итог скрининга", h2))
    story.append(_p(final.get("label", ""), callout))
    story.append(_p(
        f"Рабочий класс: {_pretty_code(final.get('class_code'))}. Согласованность слоёв: {final.get('confidence_band') or '—'}.",
        normal,
    ))
    for item in brief.get("priority", [])[:3]:
        story.append(_p("• " + item, normal))
    if brief.get("anchors_visual"):
        story.append(_table(_patient_indicator_rows({"key_facts_visual": brief.get("anchors_visual")}), [58 * mm, 45 * mm, 72 * mm], small))

    story.append(_p("1. Нормализация и контроль входных данных", h2))
    story.append(_p("Данные подготовлены к расчёту; значения, единицы и импорт остаются доступными врачу для ручной проверки.", normal))

    story.append(_p("2. Клиническая логика", h2))
    hb_sex = "женского пола" if patient.get("sex") == "F" else "мужского пола"
    if clinical.get("hemoglobin") is not None:
        story.append(_p(
            f"Гемоглобин {clinical['hemoglobin']:g} г/л; рабочий порог анемии для пациента {hb_sex} — {clinical['threshold']:g} г/л.",
            normal,
        ))
    for finding in clinical.get("findings", []):
        if finding.get("code") == "hb_threshold":
            continue
        story.append(_p("• " + finding.get("text", ""), normal))

    story.append(_p("Проверка других причин анемии", h3))
    exclusions = clinical.get("exclusions", [])
    checked = [x for x in exclusions if x.get("status") != "incomplete"]
    missing = [x for x in exclusions if x.get("status") == "incomplete"]
    status_ru = {"abnormal": "есть сигнал", "no_screening_signal": "сигнал не выявлен", "incomplete": "не хватает данных"}
    if checked:
        rows = [["Направление", "Статус", "Что видно"]]
        for item in checked:
            rows.append([item.get("title", ""), status_ru.get(item.get("status"), item.get("status", "")), item.get("text") or "—"])
        story.append(_table(rows, [55 * mm, 45 * mm, 75 * mm], small))
    if missing:
        story.append(_p("Для части дополнительных направлений данных недостаточно: " + "; ".join(f"{x.get('title')}: {', '.join(x.get('missing', []))}" for x in missing) + ".", small))

    story.append(_p("Предположительный код по МКБ-10", h3))
    story.append(_p("Код используется как справочная подсказка и не является финальным диагнозом.", small))
    if coding.get("candidates"):
        rows = [["Код", "Название", "Когда используется"]]
        for item in coding["candidates"]:
            rows.append([item.get("code", ""), item.get("label", ""), item.get("when", "")])
        story.append(_table(rows, [28 * mm, 65 * mm, 82 * mm], small))

    story.append(_p("3. ML-проверка", h2))
    if model:
        rows = [["Класс", "Модельный рейтинг / 100"]]
        for code, score in list(model.get("class_scores", {}).items())[:5]:
            rows.append([CLASSES.get(code, code), f"{float(score) * 100:.1f}"])
        story.append(_table(rows, [140 * mm, 35 * mm], small))
        sens = model.get("sensitivity") or []
        if sens:
            max_abs = max(abs(float(x.get("score_change", 0))) for x in sens) or 1.0
            rows = [["Показатель", "Значение пациента", "Влияние на модельный рейтинг / 100"]]
            for item in sens[:5]:
                strength = abs(float(item.get("score_change", 0))) / max_abs * 100
                rows.append([
                    LABS.get(item.get("feature"), {}).get("label", item.get("feature", "")),
                    f"{item.get('value', 0):g}",
                    f"{strength:.0f}",
                ])
            story.append(_table(rows, [85 * mm, 45 * mm, 45 * mm], small))
    else:
        story.append(_p("ML-слой недоступен.", normal))

    story.append(_p("4. Карта неопределённости", h2))
    if uncertainty.get("top_hypothesis"):
        top = uncertainty["top_hypothesis"]
        alt = uncertainty.get("alternative") or {}
        top_rating = float(top.get("score", 0)) * 100
        alt_rating = float(alt.get("score", 0)) * 100
        story.append(_p(
            f"{top.get('label', '')}: {top_rating:.1f}/100 ↔ {alt.get('label', '')}: {alt_rating:.1f}/100. Отрыв: {abs(top_rating-alt_rating):.1f} пункта.",
            normal,
        ))
        best = uncertainty.get("best_next_measurement")
        if best:
            story.append(_p(
                f"{best.get('label')} поможет лучше различить две ведущие гипотезы и уменьшить неопределённость диагноза.",
                callout,
            ))
        if uncertainty.get("ambiguity_factors"):
            for factor in uncertainty["ambiguity_factors"][:4]:
                story.append(_p(f"• {factor.get('factor')}: {factor.get('text')}", small))
    else:
        story.append(_p(uncertainty.get("message", "Карта неопределённости недоступна."), normal))

    story.append(_p("5. Причинно-следственные связи", h2))
    patient_comm = communication.get("causal_simple", [])
    for idx, chain in enumerate(result.get("causal_chains", [])[:3]):
        feels = patient_comm[idx].get("patient_feels", []) if idx < len(patient_comm) else []
        rows = [
            ["Что видим в анализах", "Рабочая гипотеза", "Что чувствует пациент", "Что может быть связано"],
            [
                "\n".join("• " + x for x in chain.get("markers", [])[:3]) or "—",
                chain.get("hypothesis", chain.get("label", "")),
                "\n".join("• " + x for x in feels) or "—",
                "\n".join("• " + x for x in chain.get("causes", [])[:2]) or "—",
            ],
        ]
        story.append(_table(rows, [48 * mm, 43 * mm, 40 * mm, 44 * mm], small))
        story.append(Spacer(1, 3))

    story.append(_p("6. Похожие профили", h2))
    local_rows = result.get("similar_cases", {}).get("local_confirmed", [])[:5]
    if local_rows:
        rows = [["Сходство", "Подтверждённый класс", "Диагноз врача"]]
        for item in local_rows:
            rows.append([f"{item.get('similarity', 0) * 100:.0f}%", CLASSES.get(item.get("class_code"), item.get("class_code")), item.get("confirmed_diagnosis") or "—"])
        story.append(_table(rows, [25 * mm, 82 * mm, 68 * mm], small))
    refs = result.get("similar_cases", {}).get("reference_dataset", [])[:5]
    if refs:
        story.append(_p("Похожие профили из референс-выборки", h3))
        rows = [["Сходство", "Класс", "Причина в выборке", "Возраст / пол"]]
        for item in refs:
            sex = "Ж" if item.get("sex") == "F" else "М" if item.get("sex") == "M" else str(item.get("sex") or "—")
            rows.append([
                f"{item.get('similarity', 0) * 100:.0f}%",
                CLASSES.get(item.get("class_code"), item.get("class_code")),
                _pretty_value(item.get("deficiency_cause")),
                f"{item.get('age_years', '—')} / {sex}",
            ])
        story.append(_table(rows, [22 * mm, 73 * mm, 55 * mm, 25 * mm], small))

    story.append(_p("7. ИИ-пояснение для врача", h2))
    if ai_text:
        story.extend(_markdown_flowables(ai_text, normal, small, h2, h3))
    else:
        story.append(_p("Генеративное объяснение не сформировано. Расчёт остальных слоёв от него не зависит.", normal))

    story.append(_p("Источники", h2))
    for source in result.get("sources", []):
        destination = source.get("url") or source.get("location", "")
        story.append(_p(f"{source.get('title')}. {destination}", small))

    if include_patient_page:
        story.append(PageBreak())
        story.extend(_patient_page_flowables(result, patient_name, ai_patient_text, bool(ai_patient_text), normal, small, h1, h2, h3, callout, warn, patient_style))

    doc.build(story)
    return buf.getvalue()


def _patient_page_flowables(
    result: dict[str, Any],
    patient_name: str,
    ai_patient_text: str | None,
    ai_generated: bool,
    normal,
    small,
    h1,
    h2,
    h3,
    callout,
    warn,
    patient_style,
) -> list[Any]:
    story: list[Any] = []
    final = result.get("final", {})
    communication = result.get("patient_conversation") or build_patient_conversation(result)
    uncertainty = result.get("uncertainty") or {}

    logo = _report_logo(16)
    if logo:
        story.append(logo)
        story.append(Spacer(1, 3))

    story.append(_p("Мидия · Лист пациента", h1))
    if patient_name.strip():
        story.append(_p(patient_name.strip(), h3))
        story.append(Spacer(1, 3))
    story.append(_p(
        "Этот лист помогает обсудить результат с врачом. Скрининг и текст ИИ не являются диагнозом или назначением; окончательную интерпретацию и медицинское решение выполняет врач.",
        warn,
    ))

    story.append(_p("Что показал скрининг", h2))
    story.append(_p(communication.get("plain_summary", final.get("label", "")), patient_style))

    visual_rows = _patient_indicator_rows(communication)
    if len(visual_rows) > 1:
        story.append(_p("Что видно в анализах", h2))
        story.append(_table(visual_rows, [65 * mm, 45 * mm, 65 * mm], small))

    if communication.get("causal_simple"):
        chain = communication["causal_simple"][0]
        story.append(_p("Как могут быть связаны данные", h2))
        story.append(_table([
            ["Что видим", "Рабочая гипотеза", "Что чувствует пациент", "Что может быть связано"],
            [
                "\n".join("• " + x for x in chain.get("observed_items", [])[:3]) or chain.get("observed", "—"),
                chain.get("hypothesis", "—"),
                "\n".join("• " + x for x in chain.get("patient_feels", [])[:4]) or "—",
                chain.get("possible_context", "—"),
            ],
        ], [45 * mm, 43 * mm, 42 * mm, 45 * mm], small))

    story.append(_p("Что пока требует уточнения", h2))
    story.append(_p(communication.get("uncertainty_plain", uncertainty.get("message", "")), normal))

    story.append(_p("Что поможет уточнить картину", h2))
    story.append(_p(communication.get("why_next_step", "Следующий шаг определяет врач."), callout))

    if ai_generated and ai_patient_text:
        story.append(_p("Короткое ИИ-пояснение", h2))
        story.extend(_markdown_flowables(ai_patient_text, patient_style, patient_style, h2, h3))
        # Keep the explanation card and the safety disclaimer visually separate.
        story.append(Spacer(1, 3 * mm))
        story.append(_p("ИИ-пояснение носит информационный характер и не является медицинским назначением.", warn))
    return story


def generate_patient_pdf(
    result: dict[str, Any],
    patient_name: str = "",
    ai_patient_text: str | None = None,
    ai_generated: bool = False,
) -> bytes:
    normal, small, _small_bold, h1, h2, h3, callout, warn, patient_style, *_ = _report_styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Мидия — Лист пациента",
    )
    story = _patient_page_flowables(result, patient_name, ai_patient_text, ai_generated, normal, small, h1, h2, h3, callout, warn, patient_style)
    doc.build(story)
    return buf.getvalue()
