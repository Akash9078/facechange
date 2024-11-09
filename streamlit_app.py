import streamlit as st
from PIL import Image
import io
import requests
from swapper import process_face_swap

# Configure Streamlit page
st.set_page_config(
    page_title="Face Swap App",
    page_icon="🔄",
    layout="wide"
)

def process_images(source_url: str, target_url: str):
    try:
        # Download images with headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Download source image
        source_response = requests.get(source_url, headers=headers, verify=False)
        source_response.raise_for_status()
        source_img = Image.open(io.BytesIO(source_response.content))
        
        # Download target image
        target_response = requests.get(target_url, headers=headers, verify=False)
        target_response.raise_for_status()
        target_img = Image.open(io.BytesIO(target_response.content))
        
        # Process face swap
        result = process_face_swap(source_img, target_img)
        if result:
            # Convert to bytes
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            return buf.getvalue()
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Main UI
st.title("Face Swap App")
st.write("Upload source and target image URLs to swap faces")

# Input fields
source_url = st.text_input("Source Image URL (face to use)")
target_url = st.text_input("Target Image URL (image to apply face to)")

if st.button("Swap Faces"):
    if source_url and target_url:
        with st.spinner("Processing..."):
            result_bytes = process_images(source_url, target_url)
            if result_bytes:
                st.image(result_bytes, caption="Face Swap Result")
                st.download_button(
                    label="Download Result",
                    data=result_bytes,
                    file_name="swapped_face.png",
                    mime="image/png"
                )
    else:
        st.warning("Please enter both source and target image URLs") 