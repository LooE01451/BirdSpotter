import streamlit as st
import json
import os
from streamlit_lottie import st_lottie

st.title("🌱 Digital Garden")

try:
    with open("garden_data.json", "r") as f:
        entries = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    entries = []

if entries:
    for i, entry in enumerate(entries):
        st.markdown(
            f"""
            <div style="
                background-color: #f9f9f9;
                padding: 15px 20px;
                border-radius: 10px;
                margin-bottom: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            ">
            </div>
            """, unsafe_allow_html=True
        )
        
        with st.container():
            cols = st.columns([4, 1])
            with cols[0]:
                st.image(entry["path"], width=320, caption=entry["name"])
            with cols[1]:
                st.markdown("<div style='margin-top: 60%;'></div>", unsafe_allow_html=True) #push button down
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    try:
                        if os.path.exists(entry["path"]):
                            os.remove(entry["path"])
                        entries.pop(i)
                        with open("garden_data.json", "w") as f:
                            json.dump(entries, f, indent=2)
                        st.success(f"Deleted: {entry['name']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting: {e}")

else:
    st.info("No birds identified yet.")

with open("lotties/bird1.json") as f:
    lottie = json.load(f)
st_lottie(lottie, height=300)
