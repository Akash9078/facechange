import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import requests
from PIL import Image
import io
import uuid
from contextlib import asynccontextmanager
import numpy as np
from swapper import process_face_swap  # Import the face swap function from swapper.py

# Create a temporary directory for storing processed images
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup temp files on shutdown
    for file in os.listdir(TEMP_DIR):
        os.remove(os.path.join(TEMP_DIR, file))

app = FastAPI(lifespan=lifespan)

class ImageSwapRequest(BaseModel):
    source_image_url: HttpUrl
    target_image_url: HttpUrl

def download_image(url: str) -> Image.Image:
    """Download image from URL and return PIL Image"""
    try:
        response = requests.get(str(url))
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")

@app.post("/swap-faces")
async def swap_faces(request: ImageSwapRequest):
    try:
        # Download images from URLs
        source_img = download_image(request.source_image_url)
        target_img = download_image(request.target_image_url)
        
        # Process face swap using the function from swapper.py
        result = process_face_swap(source_img, target_img)
        
        if result is None:
            raise HTTPException(status_code=400, detail="Face swap failed")
        
        # Save the result
        temp_filename = f"{TEMP_DIR}/result_{uuid.uuid4()}.png"
        result.save(temp_filename)
        
        # Return the processed image
        return FileResponse(
            temp_filename,
            media_type="image/png",
            filename="swapped_face.png",
            background=None
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}