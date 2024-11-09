import streamlit as st
from fastapi import FastAPI
import uvicorn

st.title("Face Swap API")
st.write("This is a FastAPI endpoint. Please use the /swap-faces endpoint to access the API.")
st.write("Example URL: https://facechangeapi2.streamlit.app/swap-faces") 