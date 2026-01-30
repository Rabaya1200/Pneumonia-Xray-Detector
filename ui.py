import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Pneumonia X-ray Detector")

# 1. Load the AI model directly
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5') # Make sure your model file is named model.h5

model = load_model()

# 2. Upload Image
uploaded_file = st.file_with_container("Upload an X-ray...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray.', use_column_width=True)
    
    if st.button('Run Detection'):
        # 3. Process the image and predict
        img = image.resize((224, 224)) # Change 224 to your model's size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            st.error("Result: Pneumonia Detected")
        else:
            st.success("Result: Normal / No Pneumonia")
