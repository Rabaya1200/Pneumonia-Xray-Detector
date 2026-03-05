import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# 1. Image Transformations
# This resizes images to 224x224, which ResNet expects.
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Load the Dataset from your /data folder
# Make sure your folders are: data/train/NORMAL and data/train/PNEUMONIA
train_dir = 'data/train'
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Setup the Model (ResNet18)
model = models.resnet18(weights='IMAGENET1K_V1')
# Change the last layer to have 2 outputs (Normal vs Pneumonia)
model.fc = nn.Linear(model.fc.in_features, 2)

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. Training Settings
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Fast Training Loop (Example: 1 Epoch)
print("Starting training... this may take a few minutes.")
model.train()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# 6. Save the Resulting "Brain"
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(model.state_dict(), "models/pneumonia_model.pth")
print("Success! Trained model saved as models/pneumonia_model.pth")
