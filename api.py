import streamlit as st
from PIL import Image
import io
import requests
import json
import urllib.parse
from swapper import process_face_swap

# Configure Streamlit page
st.set_page_config(
    page_title="Face Swap API",
    page_icon="🔄",
    layout="wide"
)

# API endpoint function
def process_api_request(source_url: str, target_url: str):
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

# Get query parameters
query_params = st.experimental_get_query_params()

# Check if this is an API request
if 'source_url' in query_params and 'target_url' in query_params:
    try:
        source_url = query_params['source_url'][0]
        target_url = query_params['target_url'][0]
        
        # Process the face swap
        result_bytes = process_api_request(source_url, target_url)
        
        if result_bytes:
            # Return the image as a download
            st.download_button(
                label="Download Result",
                data=result_bytes,
                file_name="swapped_face.png",
                mime="image/png"
            )
            # Also display the image
            st.image(result_bytes, caption="Face Swap Result")
        else:
            st.error("Face swap failed")
            
    except Exception as e:
        st.error(f"API Error: {str(e)}")
else:
    # Regular Streamlit UI
    st.title("Face Swap API")

    # Add API documentation
    with st.expander("API Documentation"):
        st.markdown("""
        ### API Usage
        Send a GET request to: `https://facechangeapi2.streamlit.app/?source_url=SOURCE_URL&target_url=TARGET_URL`
        
        Example curl command:
        ```bash
        curl "https://facechangeapi2.streamlit.app/?source_url=URL_TO_SOURCE_IMAGE&target_url=URL_TO_TARGET_IMAGE" \\
          -H "User-Agent: Mozilla/5.0" \\
          --output result.png
        ```
        
        Python example:
        ```python
        import requests
        import urllib.parse
        
        base_url = "https://facechangeapi2.streamlit.app/"
        source_url = urllib.parse.quote("YOUR_SOURCE_IMAGE_URL")
        target_url = urllib.parse.quote("YOUR_TARGET_IMAGE_URL")
        
        url = f"{base_url}?source_url={source_url}&target_url={target_url}"
        
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open("result.png", "wb") as f:
                f.write(response.content)
        ```
        """)

    # Regular UI inputs
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