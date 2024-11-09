import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64

def get_image_download_link(img, filename, text):
    """Generate a download link for an image"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def main():
    st.title("Face Change App")
    
    # File uploader for source image
    source_file = st.file_uploader("Upload Source Image (with face)", type=['jpg', 'jpeg', 'png'])
    
    # File uploader for target image
    target_file = st.file_uploader("Upload Target Image (to replace face)", type=['jpg', 'jpeg', 'png'])
    
    if source_file and target_file:
        # Convert uploaded files to PIL Images
        source_image = Image.open(source_file)
        target_image = Image.open(target_file)
        
        # Convert PIL images to numpy arrays for OpenCV
        source_cv = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
        target_cv = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
        
        # Display original images
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Source Image")
            st.image(source_image)
        with col2:
            st.subheader("Target Image")
            st.image(target_image)
        
        if st.button("Swap Faces"):
            try:
                # Your face swapping logic here
                # For example:
                # result = face_swap(source_cv, target_cv)
                
                # Convert result back to PIL Image
                # result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                
                # Display result
                st.subheader("Result")
                # st.image(result_image)
                
                # Add download button
                # st.markdown(get_image_download_link(result_image, "swapped_face.jpg", "Download Result"), unsafe_allow_html=True)
                
                st.success("Face swap completed successfully!")
            except Exception as e:
                st.error(f"Error during face swap: {str(e)}")

if __name__ == "__main__":
    main() 