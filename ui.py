import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

st.title("Pneumonia X-ray Detector (PyTorch)")

# 1. Load the PyTorch Brain
@st.cache_resource
def load_model():
    # Make sure your file is named 'model.pth' or 'model.pt'
    model = torch.load('model.pth', map_location=torch.device('cpu'))
    model.eval()
    return model

try:
    model = load_model()
    st.success("PyTorch Model Loaded!")
except Exception as e:
    st.error("Could not find 'model.pth' in GitHub.")

# 2. Upload and Predict
uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    # 3. Process image for PyTorch
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    if st.button("Check for Pneumonia"):
        with torch.no_grad():
            output = model(input_tensor)
            # This part depends on how your model outputs data
            prediction = torch.sigmoid(output).item() 
            
            if prediction > 0.5:
                st.error(f"Pneumonia Detected ({prediction:.2%})")
            else:
                st.success(f"Normal ({1-prediction:.2%})")
