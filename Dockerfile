FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: pre-download spaCy NER model at build time
# RUN python -m spacy download en_core_web_sm

COPY proxy.py redactor.py guard.py db.py ./

ENV SENTINEL_DB_PATH=/data/sentinel_audit.db
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "8000"]
