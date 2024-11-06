import os
import cv2
import insightface
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO

# Configure folders
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Model paths from original swapper.py
MODEL_PATHS = {
    'inswapper': {
        'path': './checkpoints/inswapper_128.onnx',
        'url': 'https://www.dropbox.com/scl/fi/h8rwajkgfrfw72w5yfbct/inswapper_128.onnx?rlkey=avqyrpfmxfxcmz8xsipsgpmg9&dl=1'
    },
    'det_10g': {
        'path': './checkpoints/models/buffalo_l/det_10g.onnx',
        'url': 'https://www.dropbox.com/scl/fi/gv67fx8vtc7phg7l7h1s5/det_10g.onnx?rlkey=wlgqbkdtrzfcg506vxpvg6n8j&dl=1'
    },
    '2d106det': {
        'path': './checkpoints/models/buffalo_l/2d106det.onnx',
        'url': 'https://www.dropbox.com/scl/fi/ly3kgdf8hg2r7eqfab4e4/2d106det.onnx?rlkey=h43adi8jnfv0he90yaatebc4k&dl=1'
    },
    '1k3d68': {
        'path': './checkpoints/models/buffalo_l/1k3d68.onnx',
        'url': 'https://www.dropbox.com/scl/fi/sj5v97t4s7s3pjmnpn97j/1k3d68.onnx?rlkey=1gnmdn93y1djl4zjomucgaeb6&dl=1'
    },
    'genderage': {
        'path': './checkpoints/models/buffalo_l/genderage.onnx',
        'url': 'https://www.dropbox.com/scl/fi/5sehilvdn13y93091trs4/genderage.onnx?rlkey=gpocnlmys0ixtkkri8dnwwsvz&dl=1'
    },
    'w600k': {
        'path': './checkpoints/models/buffalo_l/w600k_r50.onnx',
        'url': 'https://www.dropbox.com/scl/fi/a1dthaiglolxqf51gp6jb/w600k_r50.onnx?rlkey=mtafser7afgcqa7218g5s3tn3&dl=1'
    }
}

@st.cache_resource
def initialize_models():
    """Initialize face analyzer and face swapper models"""
    download_models()  # Ensure models are downloaded
    
    face_analyser = get_face_analyser()
    face_swapper = get_face_swap_model()
    
    return face_analyser, face_swapper

def download_models():
    """Download required model files if they don't exist"""
    os.makedirs('./checkpoints/models/buffalo_l', exist_ok=True)
    
    for model_name, model_info in MODEL_PATHS.items():
        if not os.path.exists(model_info['path']):
            st.info(f"Downloading {model_name} model...")
            try:
                import requests
                response = requests.get(model_info['url'])
                response.raise_for_status()
                
                with open(model_info['path'], 'wb') as f:
                    f.write(response.content)
                st.success(f"{model_name} model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading {model_name} model: {str(e)}")
                raise

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

def process_face_swap(source_img, target_img, face_analyser, face_swapper):
    try:
        # Convert PIL images to cv2 format
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        # Get face from source image
        source_face = face_analyser.get(source_img)
        if not source_face:
            raise Exception("No source face detected")
        source_face = sorted(source_face, key=lambda x: x.bbox[0])[0]

        # Get face from target image
        target_face = face_analyser.get(target_img)
        if not target_face:
            raise Exception("No target face detected")
        target_face = sorted(target_face, key=lambda x: x.bbox[0])[0]

        # Swap faces
        result = face_swapper.get(target_img, target_face, source_face, paste_back=True)
        
        # Convert back to PIL Image
        result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return result_image

    except Exception as e:
        st.error(f"Error during face swap: {str(e)}")
        return None

def main():
    st.title("FaceSwap AI")
    st.write("Upload two images to swap faces between them")

    # Initialize models
    with st.spinner("Initializing models..."):
        face_analyser, face_swapper = initialize_models()

    # File uploaders
    source_file = st.file_uploader("Choose source face image", type=['jpg', 'jpeg', 'png'])
    target_file = st.file_uploader("Choose target image", type=['jpg', 'jpeg', 'png'])

    if source_file and target_file:
        try:
            # Load images
            source_img = Image.open(source_file)
            target_img = Image.open(target_file)

            # Display original images
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Source Face")
                st.image(source_img, use_column_width=True)
            with col2:
                st.subheader("Target Image")
                st.image(target_img, use_column_width=True)

            # Process face swap
            if st.button("Swap Faces"):
                with st.spinner("Processing face swap..."):
                    result = process_face_swap(source_img, target_img, face_analyser, face_swapper)
                    
                if result:
                    st.subheader("Result")
                    st.image(result, use_column_width=True)
                    
                    # Add download button
                    buf = BytesIO()
                    result.save(buf, format="PNG")
                    st.download_button(
                        label="Download Result",
                        data=buf.getvalue(),
                        file_name="face_swap_result.png",
                        mime="image/png"
                    )

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    # Force CPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    main() 