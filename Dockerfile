FROM python:3.10-slim-bookworm

WORKDIR /app

# CPU-only ONNX Runtime configuration
ENV CUDA_VISIBLE_DEVICES=""
ENV ORT_TENSORRT_UNAVAILABLE="1"
ENV OMP_NUM_THREADS="1"
ENV OPENBLAS_NUM_THREADS="1"
ENV MKL_NUM_THREADS="1"

# Python configuration
ENV PYTHONUNBUFFERED="1"
ENV PYTHONDONTWRITEBYTECODE="1"

# Application port
ENV PORT="8000"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Download U2NET during image build.
# This prevents the application from downloading the model
# when the container starts.
RUN python -c "from rembg import new_session; new_session('u2net', providers=['CPUExecutionProvider'])"

# Copy application
COPY . .

# FastAPI listens internally on 8000
EXPOSE 8000

# Start the API
CMD ["python", "rembg_server.py"]