FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

EXPOSE 8501 8000

CMD ["python", "-m", "streamlit", "run", "HealthScreening.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
