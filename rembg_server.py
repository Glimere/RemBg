# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.responses import StreamingResponse
from rembg import remove
from io import BytesIO
from PIL import Image, ImageFilter
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
async def remove_background(
    file: UploadFile = File(...),
    binarize_threshold: int = 127,
    erode_size: int = 1
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
        
        # Use rembg to remove background
        logger.info("Running rembg background removal...")
        output_image = remove(input_image)
        
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
    logger.info("Launching server via uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
