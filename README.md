# FaceSwap AI

A web-based face swapping application built with Python Flask and InsightFace. This tool allows users to seamlessly swap faces between two images using AI.

## Features

- Easy-to-use web interface
- Drag and drop image upload
- Real-time image preview
- Download swapped images
- API endpoint support
- Supports JPG and PNG formats

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Akash9078/facechange.git
cd facechange

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Download Required Models

After cloning the repository, you need to download the required model files:

```bash
# Create checkpoints directory
mkdir -p checkpoints/models/buffalo_l

# Download the face swap model (alternative source)
wget -O ./checkpoints/inswapper_128.onnx https://huggingface.co/deepinsight/insightface/resolve/main/models/inswapper_128.onnx

# Download buffalo_l models from Hugging Face
wget -O ./checkpoints/models/buffalo_l/det_10g.onnx https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/det_10g.onnx
wget -O ./checkpoints/models/buffalo_l/2d106det.onnx https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/2d106det.onnx
wget -O ./checkpoints/models/buffalo_l/1k3d68.onnx https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/1k3d68.onnx
```

If wget doesn't work, you can manually download the files from:
- inswapper_128.onnx: https://huggingface.co/deepinsight/insightface/resolve/main/models/inswapper_128.onnx
- det_10g.onnx: https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/det_10g.onnx
- 2d106det.onnx: https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/2d106det.onnx
- 1k3d68.onnx: https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/1k3d68.onnx

Place the downloaded files in their respective directories as shown in the project structure.

## Usage

### Running the Web Interface

```bash
python swapper.py
```
Then open your browser and navigate to `http://localhost:5000`

### Using the API

You can use cURL to access the API endpoint:

```bash
curl -X POST \
     -F "source=@path/to/source.jpg" \
     -F "target=@path/to/target.jpg" \
     http://localhost:5000/api/swap \
     --output result.png
```

## Project Structure

```
facechange/
├── checkpoints/          # Directory for model files
├── templates/            # HTML templates
│   └── index.html
├── uploads/             # Temporary upload directory
├── results/             # Output directory
├── swapper.py          # Main application file
├── requirements.txt    # Python dependencies
└── README.md
```

## Dependencies

- Flask
- insightface
- OpenCV
- PIL
- onnxruntime
- numpy

## License

[MIT License](LICENSE)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

This project is built using:
- [insightface](https://github.com/deepinsight/insightface)
- [Flask](https://flask.palletsprojects.com/)