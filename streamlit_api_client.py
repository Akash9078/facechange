import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("FaceSwap AI API Client")
st.write("Enter image URLs to swap faces between them")

# API endpoint
API_URL = "http://localhost:8000/swap-face-url/"

# URL inputs
source_url = st.text_input("Enter source face image URL")
target_url = st.text_input("Enter target image URL")

def is_valid_url(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except:
        return False

def display_image_from_url(url):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    except:
        return None

if source_url and target_url:
    if is_valid_url(source_url) and is_valid_url(target_url):
        try:
            # Display original images
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Source Face")
                source_img = display_image_from_url(source_url)
                if source_img:
                    st.image(source_img, use_column_width=True)
            
            with col2:
                st.subheader("Target Image")
                target_img = display_image_from_url(target_url)
                if target_img:
                    st.image(target_img, use_column_width=True)

            # Process face swap
            if st.button("Swap Faces"):
                with st.spinner("Processing face swap..."):
                    # Prepare request data
                    data = {
                        "source_url": source_url,
                        "target_url": target_url
                    }
                    
                    # Make API request
                    response = requests.post(API_URL, json=data)
                    
                    if response.status_code == 200:
                        # Display result
                        result_image = Image.open(BytesIO(response.content))
                        st.subheader("Result")
                        st.image(result_image, use_column_width=True)
                        
                        # Add download button
                        buf = BytesIO()
                        result_image.save(buf, format="PNG")
                        st.download_button(
                            label="Download Result",
                            data=buf.getvalue(),
                            file_name="face_swap_result.png",
                            mime="image/png"
                        )
                    else:
                        st.error(f"Error: {response.text}")

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.error("Please enter valid image URLs") 