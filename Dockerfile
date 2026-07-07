FROM python:3.10-slim

WORKDIR /app

# Install system packages required by OpenCV/ONNX
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the default u2net model to avoid cold-start delays on request
RUN python -c "from rembg import new_session; new_session('u2net')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "rembg_server:app", "--host", "0.0.0.0", "--port", "8000"]
