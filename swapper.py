"It is highly built on the top of insightface, sd-webui-roop and CodeFormer."

import os
import cv2
import insightface
import numpy as np
from PIL import Image
import onnxruntime
from flask import Flask, request, send_file, render_template, jsonify
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import gdown
import requests
from tqdm import tqdm

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

MODEL_PATHS = {
    'inswapper': {
        'path': './checkpoints/inswapper_128.onnx',
        'url': 'https://drive.google.com/uc?id=1eu62GRjXWnwh-UsKKL2U0KVxA_6nRRZs'
    },
    'buffalo_l': {
        'path': './checkpoints/buffalo_l.zip',
        'url': 'https://drive.google.com/uc?id=1-joBfqA2NfMxkfW4BF5q8mwqccf1EGC1'
    }
}

def download_models():
    """Download required model files if they don't exist"""
    os.makedirs('./checkpoints', exist_ok=True)
    
    for model_name, model_info in MODEL_PATHS.items():
        if not os.path.exists(model_info['path']):
            print(f"Downloading {model_name} model...")
            try:
                gdown.download(model_info['url'], model_info['path'], quiet=False)
                
                # Extract if it's a zip file
                if model_info['path'].endswith('.zip'):
                    import zipfile
                    with zipfile.ZipFile(model_info['path'], 'r') as zip_ref:
                        zip_ref.extractall('./checkpoints')
                    # Remove zip file after extraction
                    os.remove(model_info['path'])
                
                print(f"{model_name} model downloaded successfully!")
            except Exception as e:
                print(f"Error downloading {model_name} model: {str(e)}")
                raise

def initialize_models():
    """Initialize face analyzer and face swapper models"""
    download_models()  # Ensure models are downloaded
    
    face_analyser = get_face_analyser()
    face_swapper = get_face_swap_model()
    
    return face_analyser, face_swapper

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Set environment variable to force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def get_face_swap_model():
    model_path = MODEL_PATHS['inswapper']['path']
    if not os.path.exists(model_path):
        download_models()
    return insightface.model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])

def get_face_analyser():
    if not os.path.exists('./checkpoints/buffalo_l'):
        download_models()
    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l", 
        root="./checkpoints",
        providers=['CPUExecutionProvider']
    )
    face_analyser.prepare(ctx_id=-1, det_size=(320, 320))
    return face_analyser

def process_face_swap(source_img, target_img):
    try:
        # Initialize face analyser and face swapper
        face_analyser = get_face_analyser()
        face_swapper = get_face_swap_model()

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
        print(f"Error during face swap: {str(e)}")
        return None

def process_multiple_targets(source_img, target_images):
    """Process face swap for multiple target images"""
    results = []
    for idx, target_img in enumerate(target_images):
        result = process_face_swap(source_img, target_img)
        if result:
            results.append((f'result_{idx+1}.png', result))
    return results

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/swap', methods=['POST'])
def swap_faces():
    if 'source' not in request.files or 'target' not in request.files:
        return 'No file uploaded', 400
    
    source_file = request.files['source']
    target_file = request.files['target']
    
    if source_file.filename == '' or target_file.filename == '':
        return 'No file selected', 400
    
    if not (source_file and allowed_file(source_file.filename) and 
            target_file and allowed_file(target_file.filename)):
        return 'Invalid file type', 400

    try:
        # Load images
        source_img = Image.open(source_file)
        target_img = Image.open(target_file)
        
        # Process face swap
        result = process_face_swap(source_img, target_img)
        
        if result is None:
            return 'Face swap failed', 500
        
        # Save result temporarily
        output_path = os.path.join(app.config['RESULTS_FOLDER'], 'result.png')
        result.save(output_path)
        
        # Return the processed image
        return send_file(output_path, mimetype='image/png')
    
    except Exception as e:
        return f'Error: {str(e)}', 500

@app.route('/api/swap', methods=['POST'])
def api_swap_faces():
    """API endpoint for face swapping via cURL"""
    try:
        # Check if files are in the request
        if 'source' not in request.files or 'target' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
            
        # Handle regular file uploads
        source_file = request.files['source']
        target_file = request.files['target']
        
        if source_file.filename == '' or target_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not (source_file and allowed_file(source_file.filename) and 
                target_file and allowed_file(target_file.filename)):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Load images
        source_img = Image.open(source_file)
        target_img = Image.open(target_file)

        # Process face swap
        result = process_face_swap(source_img, target_img)
        
        if result is None:
            return jsonify({'error': 'Face swap failed'}), 500
        
        # Save and return the image
        output_path = os.path.join(app.config['RESULTS_FOLDER'], 'result.png')
        result.save(output_path)
        
        return send_file(
            output_path,
            mimetype='image/png',
            as_attachment=True,
            download_name='result.png'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize models when starting the server
    try:
        print("Initializing models...")
        initialize_models()
        print("Models initialized successfully!")
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        exit(1)
    
    app.run(debug=True)