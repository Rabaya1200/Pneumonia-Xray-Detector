import streamlit as st
import requests

st.title("🫁 Pneumonia X-Ray Detector")

uploaded_file = st.file_uploader("Upload an X-ray image...", type=["jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded X-ray", width=400)

    if st.button("Run Detection"):
        files = {"file": uploaded_file.getvalue()}
        # This sends the image to our FastAPI server
        response = requests.post("http://127.0.0.1:8000/predict", files=files)

        result = response.json()["prediction"]
        if result == "Pneumonia":
            st.error(f"Prediction: {result}")
        else:
            st.success(f"Prediction: {result}")
