import streamlit as st
from PIL import Image
import uuid
import os
import json
from model_and_interface.model_interface import predict_species
from streamlit_lottie import st_lottie

st.title("📷 Bird Detection")

tab1, tab2 = st.tabs(["Upload", "About"])

with tab1:
    uploaded_file = st.file_uploader("Upload a bird photo", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Bird", use_container_width=True)
        if st.button("Identify Bird"):
            with st.spinner("Identifying bird..."):
                temp_id = str(uuid.uuid4())
                temp_path = f"images/temp_{temp_id}.jpg"
                os.makedirs("images", exist_ok=True)
                image.save(temp_path)

                bird_name = predict_species(temp_path)

                st.success(f"Identified as: **{bird_name}**")
                st.toast("🌱 Bird added to your garden!")

                final_path = f"images/{temp_id}.jpg"
                os.rename(temp_path, final_path)

                entry = {"id": temp_id, "name": bird_name, "path": final_path}
                try:
                    with open("garden_data.json", "r") as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    data = []

                data.append(entry)
                with open("garden_data.json", "w") as f:
                    json.dump(data, f)

with tab2:
    st.markdown("""
    ### About Bird Detection
    This tool uses AI to identify birds from photos.
    - 📸 Upload a clear image
    - 🧠 Model identifies the bird
    - 📄 Result added to your digital garden!
    """)
    with open("lotties/bird2.json") as f:
        lottie = json.load(f)
    st_lottie(lottie, height=250)
