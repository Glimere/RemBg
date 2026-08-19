FROM python:3.10-slim-bookworm

WORKDIR /app

# Force CPU execution, optimize single-core memory, and suppress ONNX GPU probing
ENV CUDA_VISIBLE_DEVICES=""
ENV ORT_TENSORRT_UNAVAILABLE="1"
ENV OMP_NUM_THREADS="1"
ENV PYTHONUNBUFFERED="1"
ENV PORT="8000"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models on CPUExecutionProvider to avoid runtime download delays
RUN python -c "from rembg import new_session; new_session('u2net', providers=['CPUExecutionProvider']); new_session('u2netp', providers=['CPUExecutionProvider'])"

COPY . .

EXPOSE 8000

CMD ["python", "rembg_server.py"]