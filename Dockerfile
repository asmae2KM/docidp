FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY docidp ./docidp
RUN pip install --no-deps .
COPY app.py .

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')"
# Pass keys at runtime:  docker run --env-file .env -p 8501:8501 docidp
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
