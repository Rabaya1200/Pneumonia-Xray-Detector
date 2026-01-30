import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import gdown
import os

st.title("🩺 Pneumonia X-ray Detector")

@st.cache_resource
def load_model():
    # This is your specific Google Drive ID
    file_id = '1w-yx5k2Hy6IYEag_ubAf1WBOiRmnfUhG' 
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'pneumonia_model.pth'
    
    if not os.path.exists(output):
        with st.spinner("Downloading AI Model from Google Drive..."):
            gdown.download(url, output, quiet=False)
    
    model = torch.load(output, map_location=torch.device('cpu'))
    model.eval()
    return model

try:
    model = load_model()
    st.success("✅ AI Brain Ready!")
except Exception as e:
    st.error("❌ Link Error. Make sure Google Drive sharing is set to 'Anyone with the link'.")

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    if st.button("Analyze"):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.sigmoid(output).item()
            if prediction > 0.5:
                st.error(f"Pneumonia Detected ({prediction:.2%})")
            else:
                st.success(f"Normal ({1-prediction:.2%})")
