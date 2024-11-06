# FaceSwap AI

A web-based face swapping application built with Python and Streamlit. This tool allows users to seamlessly swap faces between two images using AI.

## Features

- Easy-to-use web interface
- Drag and drop image upload
- Real-time face swapping
- Automatic model downloads
- Supports JPG and PNG formats
- CPU-based processing (no GPU required)

## System Requirements

Before installing the Python dependencies, make sure to install the required system libraries:

```bash
# On Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Or use the provided setup script
chmod +x setup.sh
./setup.sh
```

## Setup

1. Set up Python environment:

```bash
# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

## Model Files

The application will automatically download the required model files on first run. The models will be stored in the following structure:

```
project_folder/
├── checkpoints/
│   ├── inswapper_128.onnx
│   └── models/
│       └── buffalo_l/
│           ├── det_10g.onnx
│           ├── 2d106det.onnx
│           ├── 1k3d68.onnx
│           ├── genderage.onnx
│           └── w600k_r50.onnx
```

If you prefer to download the models manually, you can get them from these links:

- [inswapper_128.onnx](https://www.dropbox.com/scl/fi/h8rwajkgfrfw72w5yfbct/inswapper_128.onnx?rlkey=avqyrpfmxfxcmz8xsipsgpmg9&dl=1)
- [det_10g.onnx](https://www.dropbox.com/scl/fi/gv67fx8vtc7phg7l7h1s5/det_10g.onnx?rlkey=wlgqbkdtrzfcg506vxpvg6n8j&dl=1)
- [2d106det.onnx](https://www.dropbox.com/scl/fi/ly3kgdf8hg2r7eqfab4e4/2d106det.onnx?rlkey=h43adi8jnfv0he90yaatebc4k&dl=1)
- [1k3d68.onnx](https://www.dropbox.com/scl/fi/sj5v97t4s7s3pjmnpn97j/1k3d68.onnx?rlkey=1gnmdn93y1djl4zjomucgaeb6&dl=1)
- [genderage.onnx](https://www.dropbox.com/scl/fi/5sehilvdn13y93091trs4/genderage.onnx?rlkey=gpocnlmys0ixtkkri8dnwwsvz&dl=1)
- [w600k_r50.onnx](https://www.dropbox.com/scl/fi/a1dthaiglolxqf51gp6jb/w600k_r50.onnx?rlkey=mtafser7afgcqa7218g5s3tn3&dl=1)

## Usage

### Running the Application

```bash
streamlit run streamlit_app.py
```

The application will start and automatically open in your default web browser.

## Project Structure

```
project_folder/
├── .streamlit/           # Streamlit configuration
│   └── config.toml      
├── checkpoints/          # Model files directory
├── uploads/             # Temporary upload directory
├── results/             # Output directory
├── streamlit_app.py     # Main application file
├── requirements.txt     # Python dependencies
├── setup.sh            # Setup script
└── README.md           # Documentation
```

## Dependencies

Key dependencies include:
- Streamlit - Web interface
- InsightFace - Face analysis and swapping
- OpenCV - Image processing
- PIL - Image handling
- ONNX Runtime - Model inference
- NumPy - Numerical operations

See `requirements.txt` for a complete list of dependencies.

## Acknowledgments

This project uses:
- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis and swapping
- [Streamlit](https://streamlit.io/) - Web interface