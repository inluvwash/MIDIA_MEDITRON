"""FastAPI-интерфейс сервиса «Мидия».

Конечные точки: /health, /v1/schema, /v1/predict, /v1/import, /v1/sources,
/v1/model-info, /v1/report. Интерактивная документация: /docs (Swagger UI).
"""

from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from screening.importers import MAX_BYTES, read_upload
from screening.ml import load_bundle
from screening.report import generate_pdf
from screening.schema import CLASSES, FEATURES, LABS, VARIABLES, InputError
from screening.service import screen
from screening.sources import all_sources


class RequestSizeLimit:
    """ASGI-прослойка: отклоняет запросы тяжелее 20 МБ (HTTP 413) до обработки."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        maximum = MAX_BYTES + 65536
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except Exception:
            length = maximum + 1
        if length > maximum:
            return await JSONResponse({"detail": "Запрос превышает 20 МБ"}, status_code=413)(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > maximum:
                return await JSONResponse({"detail": "Запрос превышает 20 МБ"}, status_code=413)(scope, receive, send)
            chunks.append(body)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        return await self.app(scope, bounded_receive, send)


def authorize(x_api_key: str | None = Header(default=None)):
    """Проверка ключа из заголовка X-API-Key.

    Если переменная HEALTHSCREENING_API_KEY пуста, защита выключена (локальная демонстрация).
    Сравнение выполняется за постоянное время (hmac.compare_digest).
    """
    expected = os.getenv("HEALTHSCREENING_API_KEY", "")
    if expected and not hmac.compare_digest(expected, x_api_key or ""):
        raise HTTPException(status_code=401, detail="Требуется API key")


app = FastAPI(title="Мидия", version="5.12.1", dependencies=[Depends(authorize)])
app.add_middleware(RequestSizeLimit)


class PatientRequest(BaseModel):
    """Тело запроса: карточка пациента и признак подтверждения единиц измерения."""

    model_config = ConfigDict(extra="forbid")
    patient: dict
    units_confirmed: bool = False


class ReportRequest(PatientRequest):
    """Запрос PDF: ФИО используется только для печатной формы и не попадает в расчёт."""

    patient_name: str = ""


@app.exception_handler(InputError)
async def input_error(request, exc):
    """Ошибки входных данных (возраст, пол, единицы) возвращаются как HTTP 422."""
    return JSONResponse({"detail": str(exc)}, status_code=422)


@app.get("/health")
def health():
    """Проверка работоспособности и ограничений области применения."""
    return {
        "status": "ok",
        "version": "5.12.1",
        "minimum_age": 18,
        "pregnancy_supported": False,
        "audience": "doctor",
    }


@app.get("/v1/schema")
def schema():
    """Схема входа: допустимые лабораторные показатели, признаки модели и 12 классов."""
    return {
        "labs": LABS,
        "variables": VARIABLES,
        "model_features": FEATURES,
        "classes": CLASSES,
        "minimum_age": 18,
        "pregnancy_supported": False,
        "audience": "doctor",
    }


@app.post("/v1/predict")
def predict(body: PatientRequest):
    """Полный скрининг одного пациента (клинические правила + ML + неопределённость)."""
    return screen(body.patient, units_confirmed=body.units_confirmed)


@app.post("/v1/import")
async def import_file(file: UploadFile = File(...)):
    """Разобрать файл анализов (PDF/TXT/CSV/TSV/JSON/XLSX/XLS/HTML) в карточку пациента.

    Результат требует проверки врачом (``requires_review``) перед вызовом /v1/predict.
    """
    try:
        data = await file.read(MAX_BYTES + 1)
        patients, warnings = read_upload(data, file.filename or "")
        return {"patient": patients[0], "warnings": warnings, "requires_review": True}
    finally:
        await file.close()


@app.get("/v1/sources")
def sources():
    """Каталог источников и клинических рекомендаций."""
    return {"sources": all_sources()}


@app.get("/v1/model-info")
def model_info():
    """Метаданные модели: версия, признаки, классы, контрольная сумма."""
    try:
        _, meta = load_bundle()
        return meta
    except Exception as exc:
        return {"status": "unavailable", "detail": str(exc)}


@app.post("/v1/report")
def report(body: ReportRequest):
    """Скрининг и врачебный PDF-отчёт одним запросом."""
    result = screen(body.patient, units_confirmed=body.units_confirmed)
    return Response(
        generate_pdf(result, patient_name=body.patient_name),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="sechenov_screening_report.pdf"'},
    )
