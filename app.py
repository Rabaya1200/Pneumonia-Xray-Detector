from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io
from torchvision import models, transforms

app = FastAPI()

# 1. Load the model you just trained
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
# We load the 'brain' we just saved
model.load_state_dict(torch.load("models/pneumonia_model.pth", map_location="cpu"))
model.eval()

# 2. Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
