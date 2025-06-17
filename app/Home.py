import streamlit as st
from PIL import Image
import os



st.set_page_config(
    page_title="Bird Detector",
    page_icon="🦉",
    layout="wide",
)

logo_img = Image.open(os.path.join("logo", "logo2.jpeg"))
st.logo(logo_img, size="large")

st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🦉 Welcome to Bird Detector")

st.write("""
Welcome to the Bird Detector App!  
Use the sidebar to:
- 📷 Identify birds
- 🌱 View your digital garden
- 🏠 Learn more about this tool
""")


