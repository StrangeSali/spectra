FROM python:3.12-slim

WORKDIR /prod

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY spectra spectra
COPY setup.py setup.py
RUN pip install .

CMD ["sh", "-c", "uvicorn spectra.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
