import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


train_dataset = ImageFolder(root="H:/PythonProjects/ml-proj/ml-proj/es-ds/cifake/train", transform=transform)
val_dataset = ImageFolder(root="H:/PythonProjects/ml-proj/ml-proj/es-ds/cifake/test", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

class_labels = train_dataset.classes
print(f"Class Mapping: {class_labels}")


num_classes = 1  

# EfficientNet 
efficientnet = models.efficientnet_b3(pretrained=False)
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
efficientnet.load_state_dict(torch.load("efficientNetB3_v1.0_CIFAKE.pth", map_location=device), strict=False)
efficientnet.eval()
efficientnet.to(device)

# ResNet50
resnet50 = models.resnet50(pretrained=False)
resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)
resnet50.load_state_dict(torch.load("resNet50_v1.0_CIFAKE.pth", map_location=device))
resnet50.eval()
resnet50.to(device)

# Swin Transformer
swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
swin.head = nn.Linear(swin.head.in_features, num_classes)
swin.load_state_dict(torch.load("swin_t_v1.0_CIFAKE.pth", map_location=device))
swin.eval()
swin.to(device)



def extract_features_batch(batch_images):
    batch_images = batch_images.to(device)

    # EfficientNet 
    with torch.no_grad():
        eff_out = efficientnet(batch_images) 
    eff_prob = torch.sigmoid(eff_out) 

    # ResNet50 
    with torch.no_grad():
        res_out = resnet50(batch_images)  
    res_prob = torch.sigmoid(res_out) 

    # Swin Transformer branch
    with torch.no_grad():
        swin_out = swin(batch_images) 
    swin_prob = torch.sigmoid(swin_out) 

    # Concatenate into a (batch_size, 3) feature vector
    features = torch.cat([eff_prob, res_prob, swin_prob], dim=1)
    return features.cpu().numpy()



X_train_meta, y_train_meta = [], []

print("Starting meta training feature extraction at:", datetime.now().strftime("%H:%M:%S"))
for images, labels in train_loader:
    features = extract_features_batch(images) 
    X_train_meta.extend(features)
    y_train_meta.extend(labels.numpy())
X_train_meta = np.array(X_train_meta)
y_train_meta = np.array(y_train_meta)
print("Training meta features shape:", X_train_meta.shape)


X_val_meta, y_val_meta = [], []
print("Starting meta validation feature extraction at:", datetime.now().strftime("%H:%M:%S"))
for images, labels in val_loader:
    features = extract_features_batch(images)
    X_val_meta.extend(features)
    y_val_meta.extend(labels.numpy())
X_val_meta = np.array(X_val_meta)
y_val_meta = np.array(y_val_meta)
print("Validation meta features shape:", X_val_meta.shape)
print("Finished meta feature extraction at:", datetime.now().strftime("%H:%M:%S"))


rf_meta = RandomForestClassifier(n_estimators=100, random_state=42)
rf_meta.fit(X_train_meta, y_train_meta)

# eval on val 
y_pred = rf_meta.predict(X_val_meta)
accuracy = accuracy_score(y_val_meta, y_pred)
print(f"Random Forest Meta-Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_val_meta, y_pred, digits=4))

# save
joblib.dump(rf_meta, "meta_model_CIFAKE_FINAL.pkl")
print("Random Forest Meta-Model saved as meta_model_CIFAKE_FINAL.pkl")
