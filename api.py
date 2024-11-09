import streamlit as st
from PIL import Image
import io
import requests
from swapper import process_face_swap

# API endpoint function
def process_api_request(source_url: str, target_url: str):
    try:
        # Download images
        source_img = Image.open(io.BytesIO(requests.get(source_url).content))
        target_img = Image.open(io.BytesIO(requests.get(target_url).content))
        
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

# Streamlit interface
st.title("Face Swap API")

# Add API documentation
with st.expander("API Documentation"):
    st.markdown("""
    ### API Usage
    Send a POST request to: `https://facechangeapi2.streamlit.app/api`
    
    Request body (JSON):
    ```json
    {
        "source_url": "URL_TO_SOURCE_IMAGE",
        "target_url": "URL_TO_TARGET_IMAGE"
    }
    ```
    
    Example curl command:
    ```bash
    curl -X POST https://facechangeapi2.streamlit.app/api \\
      -H "Content-Type: application/json" \\
      -d '{"source_url":"URL_TO_SOURCE_IMAGE","target_url":"URL_TO_TARGET_IMAGE"}' \\
      --output result.png
    ```
    """)

# Regular Streamlit UI
source_url = st.text_input("Source Image URL")
target_url = st.text_input("Target Image URL")

if st.button("Swap Faces"):
    if source_url and target_url:
        with st.spinner("Processing..."):
            result_bytes = process_api_request(source_url, target_url)
            if result_bytes:
                st.image(result_bytes, caption="Face Swap Result")
                st.download_button(
                    label="Download Result",
                    data=result_bytes,
                    file_name="swapped_face.png",
                    mime="image/png"
                )