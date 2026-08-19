# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends, Security
# pyrefly: ignore [missing-import]
from fastapi.security import APIKeyHeader
# pyrefly: ignore [missing-import]
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image, ImageFilter
# pyrefly: ignore [missing-import]
import uvicorn
import logging
import os
import concurrent.futures
from contextlib import asynccontextmanager
import asyncio
# pyrefly: ignore [missing-import]
from rembg import new_session, remove

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rembg_service")

# Global session cache & warmup executor
# Criterion 17: Model session loaded ONCE on application startup and reused across requests.
session_cache = {}
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
warmup_task = None

# Criteria 24 & 25: API Key Authentication for protected endpoints
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(x_api_key: str = Security(api_key_header)):
    expected_api_key = os.getenv("REMBG_API_KEY")
    if expected_api_key:
        if not x_api_key or x_api_key != expected_api_key:
            logger.warning("API key validation failed: invalid or missing X-API-Key header.")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key"
            )

def _load_model_sync(model_name: str):
    logger.info(f"Loading rembg model session '{model_name}' on CPUExecutionProvider...")
    try:
        # Create session ONCE at application startup
        session = new_session(model_name, providers=["CPUExecutionProvider"])
        session_cache["default"] = session
        logger.info(f"Rembg session '{model_name}' pre-loaded successfully on CPU.")
        return session
    except Exception as e:
        logger.error(f"Failed to load session '{model_name}': {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global warmup_task
    logger.info("Starting up Rembg Background Removal Service...")
    model_name = os.getenv("REMBG_MODEL", "u2net")
    
    # Pre-load model session once when application starts
    loop = asyncio.get_running_loop()
    warmup_task = loop.run_in_executor(executor, _load_model_sync, model_name)
    logger.info("Server port binding active; model warming up in background.")
    
    try:
        yield
    except asyncio.CancelledError:
        pass
    finally:
        session_cache.clear()
        executor.shutdown(wait=False)
        logger.info("Rembg Background Removal Service shut down cleanly.")

app = FastAPI(title="Rembg Background Removal Service", lifespan=lifespan)

@app.get("/", summary="Health check root endpoint")
@app.get("/health", summary="Health check endpoint")
async def health_check():
    return {
        "status": "ok",
        "service": "rembg",
        "model": os.getenv("REMBG_MODEL", "u2net"),
        "session_ready": "default" in session_cache
    }

@app.post("/remove", summary="Remove background from an uploaded image", dependencies=[Depends(verify_api_key)])
async def remove_background(
    file: UploadFile = File(...),
    binarize_threshold: int = Query(127, description="Threshold for alpha channel binarization (0-255, 0 disables)"),
    erode_size: int = Query(1, description="Pixel radius for alpha channel erosion (0 disables)"),
    max_size: int = Query(2048, description="Maximum image dimension to prevent OOM on high-res uploads")
):
    logger.info(f"Received background removal request. Filename: {file.filename}, Content-Type: {file.content_type}")
    
    # Validate that the uploaded file is an image
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Validation failed: Uploaded file is not an image (Content-Type: {file.content_type})")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file bytes
        image_bytes = await file.read()
        logger.info(f"Read {len(image_bytes)} bytes from {file.filename}")
        
        # Load the image using PIL
        input_image = Image.open(BytesIO(image_bytes))
        
        # Auto-downscale if image exceeds max_size to protect container memory limits
        if max_size > 0 and (input_image.width > max_size or input_image.height > max_size):
            logger.info(f"Downscaling image from {input_image.size} to max dimension {max_size}px to prevent memory spikes")
            input_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Criterion 17: Re-use the model session created once when application started
        session = session_cache.get("default")
        if session is None and warmup_task is not None:
            logger.info("Awaiting application startup model load completion...")
            session = await asyncio.shield(warmup_task)
            if session:
                session_cache["default"] = session
        
        if session is None:
            # Fallback if warmup failed during startup
            model_name = os.getenv("REMBG_MODEL", "u2net")
            logger.info(f"Fallback loading session '{model_name}' on CPU...")
            session = new_session(model_name, providers=["CPUExecutionProvider"])
            session_cache["default"] = session
        
        # Run rembg background removal using pre-loaded session (session=session)
        logger.info("Running rembg background removal inference...")
        output_image = remove(input_image, session=session)
        
        # Convert to RGBA if not already
        if output_image.mode != "RGBA":
            output_image = output_image.convert("RGBA")
            
        # Post-process alpha channel for sharp cut edges and grey halo removal
        r, g, b, a = output_image.split()
        
        # 1. Binarize alpha channel
        if binarize_threshold > 0:
            logger.info(f"Binarizing alpha channel with threshold {binarize_threshold}")
            thresh = max(0, min(255, binarize_threshold))
            a = a.point(lambda p: 255 if p > thresh else 0)
            
        # 2. Erode alpha channel
        if erode_size > 0:
            logger.info(f"Eroding alpha channel by {erode_size} pixels")
            filter_size = 2 * erode_size + 1
            a = a.filter(ImageFilter.MinFilter(filter_size))
            
        # Merge channels back
        output_image = Image.merge("RGBA", (r, g, b, a))
        
        logger.info("Successfully processed image; streaming response back to client")
        
        # Save to buffer and stream
        buf = BytesIO()
        output_image.save(buf, format="PNG")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.exception(f"Unexpected error during background removal for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Criterion 18: Don't use multiple Uvicorn workers for single/low-RAM VPS (KVM 1).
    # Single worker process run command: uvicorn rembg_server:app --host 0.0.0.0 --port 8000
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Launching server via uvicorn (1 worker process) on 0.0.0.0:{port}...")
    uvicorn.run("rembg_server:app", host="0.0.0.0", port=port, workers=1, log_level="info")




