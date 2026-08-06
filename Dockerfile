FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from rembg import new_session; new_session('u2net')"

COPY . .

EXPOSE 8000

CMD sh -c "uvicorn rembg_server:app --host 0.0.0.0 --port ${PORT:-8000}"