from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl
import cv2
import numpy as np
import urllib.request
from PIL import Image
import io
import uuid
import os

app = FastAPI()

# Create a temporary directory for storing processed images
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

class ImageSwapRequest(BaseModel):
    source_image_url: HttpUrl  # URL of the source image
    target_image_url: HttpUrl  # URL of the target image

def url_to_cv2(url):
    """Convert image URL to CV2 format"""
    try:
        # Download the image from URL
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to load image from URL")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image from URL: {str(e)}")

def detect_face(image):
    """Detect face in an image and return the face landmarks"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    return faces[0]

def face_swap(source_img, target_img):
    """Perform basic face swapping"""
    # Detect faces in both images
    source_face = detect_face(source_img)
    target_face = detect_face(target_img)
    
    # Extract face region from source
    x, y, w, h = source_face
    source_face_img = source_img[y:y+h, x:x+w]
    
    # Extract face region from target
    x2, y2, w2, h2 = target_face
    
    # Resize source face to match target face size
    source_face_img = cv2.resize(source_face_img, (w2, h2))
    
    # Create a copy of target image
    result = target_img.copy()
    
    # Replace the face region
    result[y2:y2+h2, x2:x2+w2] = source_face_img
    
    return result

@app.post("/swap-faces")
async def swap_faces(request: ImageSwapRequest):
    try:
        # Convert URLs to CV2 format
        source_cv = url_to_cv2(str(request.source_image_url))
        target_cv = url_to_cv2(str(request.target_image_url))
        
        # Perform face swapping
        result = face_swap(source_cv, target_cv)
        
        # Save the result to a temporary file
        temp_filename = f"{TEMP_DIR}/result_{uuid.uuid4()}.jpg"
        cv2.imwrite(temp_filename, result)
        
        # Return the file
        return FileResponse(
            temp_filename,
            media_type="image/jpeg",
            filename="swapped_face.jpg",
            background=None
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Optional: Cleanup endpoint to remove temporary files
@app.on_event("shutdown")
async def cleanup():
    for file in os.listdir(TEMP_DIR):
        os.remove(os.path.join(TEMP_DIR, file))