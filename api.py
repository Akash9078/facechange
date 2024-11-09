import os
import cv2
import insightface
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from io import BytesIO
import requests
from pydantic import BaseModel, HttpUrl

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Configure folders
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = FastAPI(
    title="FaceSwap API",
    description="API for swapping faces in images using insightface",
    version="1.0.0"
)

# Reuse the MODEL_PATHS from streamlit_app.py
MODEL_PATHS = {
    'inswapper': {
        'path': './checkpoints/inswapper_128.onnx',
        'url': 'https://www.dropbox.com/scl/fi/h8rwajkgfrfw72w5yfbct/inswapper_128.onnx?rlkey=avqyrpfmxfxcmz8xsipsgpmg9&dl=1'
    },
    'det_10g': {
        'path': './checkpoints/models/buffalo_l/det_10g.onnx',
        'url': 'https://www.dropbox.com/scl/fi/gv67fx8vtc7phg7l7h1s5/det_10g.onnx?rlkey=wlgqbkdtrzfcg506vxpvg6n8j&dl=1'
    },
    'w600k': {
        'path': './checkpoints/models/buffalo_l/w600k_r50.onnx',
        'url': 'https://www.dropbox.com/scl/fi/a1dthaiglolxqf51gp6jb/w600k_r50.onnx?rlkey=mtafser7afgcqa7218g5s3tn3&dl=1'
    }
}

def download_models():
    """Download required model files if they don't exist"""
    os.makedirs('./checkpoints/models/buffalo_l', exist_ok=True)
    
    for model_name, model_info in MODEL_PATHS.items():
        if not os.path.exists(model_info['path']):
            try:
                import requests
                response = requests.get(model_info['url'])
                response.raise_for_status()
                
                with open(model_info['path'], 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error downloading {model_name} model: {str(e)}")

def get_face_swap_model():
    model_path = MODEL_PATHS['inswapper']['path']
    return insightface.model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])

def get_face_analyser():
    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l", 
        root="./checkpoints",
        providers=['CPUExecutionProvider']
    )
    face_analyser.prepare(ctx_id=-1, det_size=(320, 320))
    return face_analyser

# Initialize models at startup
download_models()
face_analyser = get_face_analyser()
face_swapper = get_face_swap_model()

def process_face_swap(source_img, target_img):
    try:
        # Convert PIL images to cv2 format
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        # Get face from source image
        source_face = face_analyser.get(source_img)
        if not source_face:
            raise HTTPException(status_code=400, detail="No source face detected")
        source_face = sorted(source_face, key=lambda x: x.bbox[0])[0]

        # Get face from target image
        target_face = face_analyser.get(target_img)
        if not target_face:
            raise HTTPException(status_code=400, detail="No target face detected")
        target_face = sorted(target_face, key=lambda x: x.bbox[0])[0]

        # Swap faces
        result = face_swapper.get(target_img, target_face, source_face, paste_back=True)
        
        # Convert back to PIL Image
        result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return result_image

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during face swap: {str(e)}")

class SwapRequest(BaseModel):
    source_url: HttpUrl
    target_url: HttpUrl

def download_image(url: str) -> Image.Image:
    """Download image from URL and return PIL Image"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {str(e)}")

@app.post("/swap-face-url/")
async def swap_face_url(request: SwapRequest):
    """
    Swap faces between two images using URLs
    - source_url: URL of image containing the face to be swapped
    - target_url: URL of image where the face will be placed
    """
    try:
        # Download images from URLs
        source_img = download_image(str(request.source_url))
        target_img = download_image(str(request.target_url))
        
        # Process face swap
        result = process_face_swap(source_img, target_img)
        
        # Convert result to bytes
        img_byte_arr = BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(content=img_byte_arr, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 