from pathlib import Path

import pandas as pd
import pytest

from screening.coding import coding_candidates
from screening.importers import read_upload
from screening.report import generate_pdf
from screening.schema import CLASS_ORDER, CLASSES, VARIABLES, InputError, normalize_record
from screening.service import screen

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_contract():
    frame = pd.read_csv(ROOT / "data" / "deficiency_anemia.csv")
    assert frame.shape == (840, 48)
    assert frame.age_years.min() == 18
    assert frame.age_years.max() == 93
    assert int((frame.age_years < 18).sum()) == 0
    assert set(frame.anemia_class) == set(CLASS_ORDER)


def test_variables_xlsx_contract():
    frame = pd.read_csv(ROOT / "data" / "deficiency_anemia.csv", nrows=1)
    variable_names = [x["variable"] for x in VARIABLES]
    assert len(variable_names) == 48
    assert variable_names == list(frame.columns)
    assert sum(1 for x in VARIABLES if x["role"] == "laboratory_feature") == 35


def test_exactly_twelve_target_classes():
    assert len(CLASSES) == 12
    result = screen(
        {"age_years": 40, "sex": "F", "hemoglobin": 119, "ferritin": 12, "TSAT": 10},
        units_confirmed=True,
    )
    assert result["final"]["class_code"] in CLASSES
    assert "group_code" not in result["final"]
    assert "group_label" not in result["final"]


def test_under18_rejected():
    with pytest.raises(InputError):
        normalize_record(
            {"age_years": 17, "sex": "F", "hemoglobin": {"value": 120, "unit": "г/л"}},
            units_confirmed=True,
        )


def test_pregnancy_rejected():
    with pytest.raises(InputError):
        normalize_record(
            {"age_years": 28, "sex": "F", "pregnant": True, "hemoglobin": {"value": 110, "unit": "г/л"}},
            units_confirmed=True,
        )


def test_screen_smoke_and_layers():
    raw = {
        "age_years": 35,
        "sex": "F",
        "hemoglobin": {"value": 110, "unit": "г/л"},
        "ferritin": {"value": 12, "unit": "мкг/л"},
        "TSAT": {"value": 10, "unit": "%"},
        "Ret_He": {"value": 25, "unit": "пг"},
        "CRP": {"value": 2, "unit": "мг/л"},
        "complaints": "слабость",
        "symptoms": ["Утомляемость / слабость"],
        "history": "обильные менструации последние 6 месяцев",
    }
    result = screen(raw, units_confirmed=True)
    assert result["schema_version"] == 8
    assert result["clinical"]["anemia"] is True
    assert result["final"]["class_code"] in result["model"]["class_scores"]
    assert result["uncertainty"]["status"] in {"low", "moderate", "high"}
    assert result["causal_chains"]
    assert result["doctor_brief"]["priority"]
    assert result["patient_conversation"]["plain_summary"]
    assert result["coding"]["status"] == "reference_only"


def test_free_text_anamnesis_affects_causal_context_not_ml_features():
    raw = {
        "age_years": 35,
        "sex": "F",
        "hemoglobin": {"value": 108, "unit": "г/л"},
        "ferritin": {"value": 10, "unit": "мкг/л"},
        "TSAT": {"value": 9, "unit": "%"},
        "history": "обильные менструации с хронической кровопотерей",
    }
    result = screen(raw, units_confirmed=True)
    causal_text = " ".join(" ".join(x.get("causes", [])) for x in result["causal_chains"])
    assert "кровопот" in causal_text.lower() or "менстру" in causal_text.lower()
    assert "history" not in result["model"].get("features_used", [])


def test_uncertainty_has_specific_next_test_and_reason_map():
    raw = {
        "age_years": 61,
        "sex": "F",
        "hemoglobin": {"value": 108, "unit": "г/л"},
        "ferritin": {"value": 55, "unit": "мкг/л"},
        "CRP": {"value": 18, "unit": "мг/л"},
        "TSAT": {"value": 14, "unit": "%"},
        "MCV": {"value": 84, "unit": "фл"},
    }
    result = screen(raw, units_confirmed=True)
    best = result["uncertainty"].get("best_next_measurement")
    assert best is not None
    assert best.get("label")
    assert best.get("why")


def test_who_ferritin_inflammation_rule_is_visible():
    raw = {
        "age_years": 50,
        "sex": "M",
        "hemoglobin": {"value": 115, "unit": "г/л"},
        "ferritin": {"value": 55, "unit": "мкг/л"},
        "CRP": {"value": 20, "unit": "мг/л"},
        "TSAT": {"value": 14, "unit": "%"},
    }
    result = screen(raw, units_confirmed=True)
    texts = " ".join(x["text"] for x in result["clinical"]["findings"])
    assert "< 70" in texts
    assert any(s["id"] == "who_ferritin_2020" for s in result["sources"])


def test_icd_is_reference_not_autocode():
    item = coding_candidates("iron_deficiency_anemia")
    assert item["status"] == "reference_only"
    assert any(c["code"] == "D50" for c in item["candidates"])
    assert "не" in item["note"].lower()


def test_single_patient_import_and_multirow_rejection():
    one = "age_years,sex,hemoglobin,ferritin\n40,F,118,16\n".encode("utf-8")
    records, warnings = read_upload(one, "one.csv")
    assert len(records) == 1
    assert warnings

    many = "age_years,sex,hemoglobin\n40,F,118\n52,M,125\n".encode("utf-8")
    with pytest.raises(ValueError, match="несколько пациентов"):
        read_upload(many, "many.csv")


def test_fio_is_ephemeral_report_only():
    raw = {
        "age_years": 50,
        "sex": "M",
        "hemoglobin": {"value": 120, "unit": "г/л"},
        "ferritin": {"value": 20, "unit": "мкг/л"},
        "TSAT": {"value": 12, "unit": "%"},
    }
    result = screen(raw, units_confirmed=True)
    assert "full_name" not in result["patient"]
    data = generate_pdf(result, patient_name="Иванов Иван Иванович")
    assert data[:4] == b"%PDF"
    assert len(data) > 7000


def test_no_persistent_history_module_in_runtime_contract():
    assert not (ROOT / "data" / "history.db").exists()
    assert not (ROOT / "data" / ".history_salt").exists()


def test_pdf_import_prefills_patient_card_fields(tmp_path):
    from io import BytesIO
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf)
    lines = [
        "Patient: Ivan Ivanov",
        "Age: 44",
        "Sex: male",
        "Complaints: fatigue and dizziness",
        "Symptoms: shortness of breath",
        "History: chronic gastritis",
        "Medications: metformin",
        "Hemoglobin 118",
        "Ferritin 18",
    ]
    y = 800
    for line in lines:
        c.drawString(50, y, line)
        y -= 22
    c.save()

    records, warnings = read_upload(buf.getvalue(), "patient.pdf")
    rec = records[0]
    assert rec["full_name"] == "Ivan Ivanov"
    assert rec["age_years"] == 44
    assert str(rec["sex"]).lower() == "male"
    complaints = rec.get("complaints", "").lower()
    assert "fatigue" in complaints
    assert "shortness of breath" in complaints
    assert "gastritis" in rec.get("history", "").lower()
    assert "metformin" in rec.get("medications", "").lower()
    assert "patient_priority" not in rec
    assert "hemoglobin" in rec and "ferritin" in rec
    assert any("карточ" in x.lower() for x in warnings)


def test_local_confirmed_casebase_is_anonymous_and_deletable(tmp_path, monkeypatch):
    import screening.casebase as casebase

    monkeypatch.setattr(casebase, "CASE_PATH", tmp_path / "confirmed_cases.json")
    patient = normalize_record(
        {
            "age_years": 50,
            "sex": "M",
            "hemoglobin": {"value": 120, "unit": "г/л"},
            "ferritin": {"value": 20, "unit": "мкг/л"},
            "TSAT": {"value": 12, "unit": "%"},
            "complaints": "слабость",
        },
        units_confirmed=True,
    )
    row = casebase.add_case(patient, "iron_deficiency_anemia", "ЖДА", "D50")
    stored = casebase.list_cases()
    assert len(stored) == 1
    assert stored[0]["case_id"] == row["case_id"]
    assert "full_name" not in stored[0]
    assert "context" not in stored[0]
    assert casebase.delete_case(row["case_id"]) is True
    assert casebase.list_cases() == []




def test_plain_txt_import_prefills_card_and_labs_without_table_delimiter():
    txt = """ФИО: Петров Петр Петрович
Возраст: 51
Пол: мужской
Жалобы при обращении: слабость, головокружение
Дата обращения: 04.10.2026
Анамнез: хронический гастрит
Номер заказа: 123456
Постоянные препараты: метформин
Гемоглобин: 116 г/л
Ферритин: 17 мкг/л
Насыщение трансферрина: 11 %
""".encode("utf-8")
    records, warnings = read_upload(txt, "patient.txt")
    rec = records[0]
    assert rec["full_name"] == "Петров Петр Петрович"
    assert rec["age_years"] == 51
    assert "слабость" in rec.get("complaints", "").lower()
    assert "дата обращения" not in rec.get("complaints", "").lower()
    assert "номер заказа" not in rec.get("history", "").lower()
    assert "метформин" in rec.get("medications", "").lower()
    assert "hemoglobin" in rec and "ferritin" in rec and "TSAT" in rec
    assert any("свобод" in x.lower() or "пров" in x.lower() for x in warnings)


def test_ai_markdown_is_rendered_without_raw_hash_headings_in_pdf():
    from io import BytesIO
    from pypdf import PdfReader

    raw = {
        "age_years": 50,
        "sex": "M",
        "hemoglobin": {"value": 120, "unit": "г/л"},
        "ferritin": {"value": 20, "unit": "мкг/л"},
        "TSAT": {"value": 12, "unit": "%"},
    }
    result = screen(raw, units_confirmed=True)
    data = generate_pdf(
        result,
        patient_name="Иванов Иван Иванович",
        ai_text="# Заголовок\n\n**Ключевое:** текст\n\n- пункт один\n- пункт два",
        ai_patient_text="Скрининг выявил изменения, которые нужно обсудить с врачом.",
    )
    text = "\n".join((p.extract_text() or "") for p in PdfReader(BytesIO(data)).pages)
    assert "# Заголовок" not in text
    assert "Заголовок" in text
    assert "Короткое" not in text or "пациент" in text.lower()
    assert "Скрининг выявил изменения" in text


def test_pdf_style_flattened_sections_and_fio_patient_are_separated():
    txt = (
        "ФИО пациента: Сидорова Анна Сергеевна   Возраст: 42   Пол: женский\n"
        "Жалобы: слабость, головокружение Анамнез: хронический гастрит "
        "Принимала: железо в таблетках последние 3 месяца\n"
        "Дата обращения: 04.10.2026 Номер заказа: 12345\n"
        "Гемоглобин: 112 г/л Ферритин: 14 мкг/л Насыщение трансферрина: 10 %\n"
    ).encode("utf-8")
    records, _ = read_upload(txt, "flattened.txt")
    rec = records[0]
    assert rec["full_name"] == "Сидорова Анна Сергеевна"
    assert rec["age_years"] == 42
    assert "слабость" in rec.get("complaints", "").lower()
    assert "анамнез" not in rec.get("complaints", "").lower()
    assert "хронический гастрит" in rec.get("history", "").lower()
    assert "принимала" not in rec.get("history", "").lower()
    assert "железо" in rec.get("medications", "").lower()
    assert "дата обращения" not in rec.get("medications", "").lower()


def test_fio_can_be_assembled_from_separate_name_fields():
    txt = """Фамилия: Орлова
Имя: Мария
Отчество: Игоревна
Возраст: 39
Пол: женский
Жалобы: утомляемость
Гемоглобин: 115 г/л
Ферритин: 16 мкг/л
""".encode("utf-8")
    records, _ = read_upload(txt, "separate_name_fields.txt")
    assert records[0]["full_name"] == "Орлова Мария Игоревна"


def test_ai_splitter_keeps_patient_text_out_of_doctor_part():
    from screening.rag import _split_generated_text

    result = screen(
        {"age_years": 45, "sex": "F", "hemoglobin": 115, "ferritin": 15, "TSAT": 11},
        units_confirmed=True,
    )
    generated = (
        "### **ДЛЯ ВРАЧА:**\n"
        "Профессиональное объяснение врачу.\n\n"
        "**ДЛЯ ПАЦИЕНТА:**\n"
        "Простое объяснение пациенту."
    )
    doctor, patient = _split_generated_text(generated, result)
    assert "Простое объяснение пациенту" not in doctor
    assert "Профессиональное объяснение врачу" in doctor
    assert "Простое объяснение пациенту" in patient


def test_report_card_is_not_numbered_as_layer_and_layers_start_from_normalization():
    from io import BytesIO
    from pypdf import PdfReader

    result = screen(
        {"age_years": 45, "sex": "F", "hemoglobin": 115, "ferritin": 15, "TSAT": 11},
        units_confirmed=True,
    )
    data = generate_pdf(
        result,
        patient_name="Сидорова Анна Сергеевна",
        ai_text="Пояснение врачу.",
        ai_patient_text="Пояснение пациенту.",
    )
    pdf_text = "\n".join((page.extract_text() or "") for page in PdfReader(BytesIO(data)).pages)
    assert "1. Карточка пациента" not in pdf_text
    assert "Карточка пациента" in pdf_text
    assert "1. Нормализация и контроль входных данных" in pdf_text
    assert "2. Клиническая логика" in pdf_text
    assert "7. ИИ-пояснение для врача" in pdf_text
