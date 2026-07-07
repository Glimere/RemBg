# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.responses import StreamingResponse
from rembg import remove
from io import BytesIO
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import uvicorn
import logging
from contextlib import asynccontextmanager
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rembg_service")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Rembg Background Removal Service...")
    try:
        yield
    except asyncio.CancelledError:
        # Suppress the harmless CancelledError thrown when pressing CTRL+C
        pass
    finally:
        logger.info("Rembg Background Removal Service shut down cleanly.")

app = FastAPI(title="Rembg Background Removal Service", lifespan=lifespan)

@app.post("/remove", summary="Remove background from an uploaded image")
async def remove_background(file: UploadFile = File(...)):
    logger.info(f"Received background removal request. Filename: {file.filename}, Content-Type: {file.content_type}")
    
    # Validate that the uploaded file is an image
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning(f"Validation failed: Uploaded file is not an image (Content-Type: {file.content_type})")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file bytes
        image_bytes = await file.read()
        logger.info(f"Read {len(image_bytes)} bytes from {file.filename}")
        
        # Use rembg to remove background directly from the raw bytes
        logger.info("Running rembg background removal...")
        output_bytes = remove(image_bytes)
        logger.info(f"Background removal complete. Generated {len(output_bytes)} bytes")
        
        logger.info("Successfully processed image; streaming response back to client")
        return StreamingResponse(BytesIO(output_bytes), media_type="image/png")
    except Exception as e:
        logger.exception(f"Unexpected error during background removal for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Launching server via uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
