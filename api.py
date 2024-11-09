import streamlit as st
import requests
from PIL import Image
import io
from swapper import process_face_swap

st.title("Face Swap API")

def download_image(url: str) -> Image.Image:
    """Download image from URL and return PIL Image"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Error downloading image: {str(e)}")
        return None

def main():
    # Input fields for image URLs
    source_url = st.text_input("Source Image URL")
    target_url = st.text_input("Target Image URL")
    
    if st.button("Swap Faces"):
        if source_url and target_url:
            try:
                # Download images
                with st.spinner("Downloading images..."):
                    source_img = download_image(source_url)
                    target_img = download_image(target_url)
                
                if source_img and target_img:
                    # Process face swap
                    with st.spinner("Processing face swap..."):
                        result = process_face_swap(source_img, target_img)
                        
                        if result:
                            # Display result
                            st.image(result, caption="Face Swap Result")
                            
                            # Add download button
                            buf = io.BytesIO()
                            result.save(buf, format="PNG")
                            st.download_button(
                                label="Download Result",
                                data=buf.getvalue(),
                                file_name="swapped_face.png",
                                mime="image/png"
                            )
                        else:
                            st.error("Face swap failed")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter both source and target image URLs")

if __name__ == "__main__":
    main()