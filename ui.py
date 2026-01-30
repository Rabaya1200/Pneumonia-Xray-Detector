import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Pneumonia X-ray Detector")

# 1. Load the AI brain directly from your GitHub folder
@st.cache_resource
def load_my_model():
    # Make sure you have a file named 'model.h5' in your GitHub!
    return tf.keras.models.load_model('model.h5')

try:
    model = load_my_model()
    st.success("AI Model Loaded Successfully!")
except Exception as e:
    st.error("Missing 'model.h5' file in GitHub. Please upload it!")

# 2. Upload Image
uploaded_file = st.file_uploader("Upload an X-ray...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray.', use_container_width=True)
    
    if st.button('Run Detection'):
        # 3. Predict immediately without calling another server
        img = image.resize((224, 224)) 
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            st.error("Result: Pneumonia Detected")
        else:
            st.success("Result: Normal / No Pneumonia")
