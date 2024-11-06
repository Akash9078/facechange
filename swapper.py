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
import requests
from tqdm import tqdm
import socket
from contextlib import closing

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

def download_models():
    """Download required model files if they don't exist"""
    os.makedirs('./checkpoints/models/buffalo_l', exist_ok=True)
    
    for model_name, model_info in MODEL_PATHS.items():
        if not os.path.exists(model_info['path']):
            print(f"Downloading {model_name} model...")
            try:
                response = requests.get(model_info['url'], stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024
                progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

                with open(model_info['path'], 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                
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

def find_available_port(start_port, max_attempts=100):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            try:
                sock.bind(('0.0.0.0', port))
                return port
            except socket.error:
                continue
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

if __name__ == '__main__':
    # Initialize models when starting the server
    try:
        print("Initializing models...")
        initialize_models()
        print("Models initialized successfully!")
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        exit(1)
    
    # Get preferred port from environment variable or use 6000 as default
    preferred_port = int(os.environ.get('PORT', 6000))
    
    try:
        # Try to find an available port starting from the preferred port
        port = find_available_port(preferred_port)
        if port != preferred_port:
            print(f"Port {preferred_port} is in use, using port {port} instead")
        print(f"Server starting on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        exit(1)