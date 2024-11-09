import streamlit as st
from PIL import Image
import io
from swapper import process_face_swap

# Configure Streamlit page
st.set_page_config(
    page_title="Face Swap App",
    page_icon="🔄",
    layout="wide"
)

def process_images(source_img: Image.Image, target_img: Image.Image):
    try:
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
st.write("Upload source and target images to swap faces")

# File upload fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Source Image (face to use)")
    source_file = st.file_uploader("Choose source image", type=['jpg', 'jpeg', 'png'])
    if source_file:
        source_img = Image.open(source_file)
        st.image(source_file, caption="Source Image", use_column_width=True)

with col2:
    st.subheader("Target Image (image to apply face to)")
    target_file = st.file_uploader("Choose target image", type=['jpg', 'jpeg', 'png'])
    if target_file:
        target_img = Image.open(target_file)
        st.image(target_file, caption="Target Image", use_column_width=True)

if st.button("Swap Faces"):
    if source_file is not None and target_file is not None:
        with st.spinner("Processing..."):
            result_bytes = process_images(source_img, target_img)
            if result_bytes:
                st.success("Face swap completed!")
                st.image(result_bytes, caption="Face Swap Result")
                st.download_button(
                    label="Download Result",
                    data=result_bytes,
                    file_name="swapped_face.png",
                    mime="image/png"
                )
    else:
        st.warning("Please upload both source and target images")

# Add information about supported file types
st.sidebar.markdown("""
### Supported File Types
- JPG/JPEG
- PNG

### Tips
1. Make sure faces are clearly visible
2. Good lighting helps
3. Front-facing photos work best
""")