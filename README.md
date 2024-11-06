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

After cloning the repository, you need to download the required model files. You can download them manually from these Dropbox links:

- [inswapper_128.onnx](https://www.dropbox.com/scl/fi/h8rwajkgfrfw72w5yfbct/inswapper_128.onnx?rlkey=avqyrpfmxfxcmz8xsipsgpmg9&dl=1)
- [det_10g.onnx](https://www.dropbox.com/scl/fi/gv67fx8vtc7phg7l7h1s5/det_10g.onnx?rlkey=wlgqbkdtrzfcg506vxpvg6n8j&dl=1)
- [2d106det.onnx](https://www.dropbox.com/scl/fi/ly3kgdf8hg2r7eqfab4e4/2d106det.onnx?rlkey=h43adi8jnfv0he90yaatebc4k&dl=1)
- [1k3d68.onnx](https://www.dropbox.com/scl/fi/sj5v97t4s7s3pjmnpn97j/1k3d68.onnx?rlkey=1gnmdn93y1djl4zjomucgaeb6&dl=1)
- [genderage.onnx](https://www.dropbox.com/scl/fi/5sehilvdn13y93091trs4/genderage.onnx?rlkey=gpocnlmys0ixtkkri8dnwwsvz&dl=1)
- [w600k_r50.onnx](https://www.dropbox.com/scl/fi/a1dthaiglolxqf51gp6jb/w600k_r50.onnx?rlkey=mtafser7afgcqa7218g5s3tn3&dl=1)

Place the downloaded files in the following structure:
```
facechange/
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

## Usage

### Running the Web Interface

```bash
python swapper.py
```
Then open your browser and navigate to `http://localhost:6000`

### Using the API

You can use cURL to access the API endpoint:

```bash
curl -X POST \
     -F "source=@path/to/source.jpg" \
     -F "target=@path/to/target.jpg" \
     http://localhost:6000/api/swap \
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