FROM python:3.12-slim

# 1. Install system dependencies + C/C++ compilers required for package builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /prod

# 2. Set an absolute path for TF Hub models so they persist inside the image
ENV TFHUB_CACHE_DIR=/prod/tfhub_modules

COPY requirements.txt requirements.txt

# 3. Upgrade pip & explicitly install setuptools so 'pkg_resources' is available
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY spectra spectra
COPY setup.py setup.py
RUN pip install .

# 4. Pre-cache YAMNet using the COMPLETE URL path during image build
RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1')"

CMD ["sh", "-c", "uvicorn spectra.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
