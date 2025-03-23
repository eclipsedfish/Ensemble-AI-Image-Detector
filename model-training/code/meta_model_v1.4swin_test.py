import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import models, transforms
from PIL import Image
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])



# EfficientNet
efficientnet = models.efficientnet_b3(pretrained=False)
num_classes = 1  
efficientnet.classifier[1] = torch.nn.Linear(efficientnet.classifier[1].in_features, num_classes)
efficientnet.load_state_dict(torch.load("efficientNetB3_v1.0_CIFAKE.pth", map_location=device), strict=False)
efficientnet.eval()
efficientnet.to(device)

# ResNet50 
resnet50 = models.resnet50(pretrained=True)
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)
resnet50.load_state_dict(torch.load("resNet50_v1.0_CIFAKE.pth", map_location=device))
resnet50.to(device)
resnet50.eval()

# Swin 
swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
swin.head = torch.nn.Linear(swin.head.in_features, 1)
swin.load_state_dict(torch.load("swin_t_v1.0_CIFAKE.pth", map_location=device))
swin.eval()
swin.to(device)


meta_model = joblib.load("meta_model_CIFAKE_FINAL.pkl")
class_labels = ["FAKE", "REAL"]


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

    # Swin Transformer 
    with torch.no_grad():
        swin_out = swin(batch_images)  
    swin_prob = torch.sigmoid(swin_out)  

    features = torch.cat([eff_prob, res_prob, swin_prob], dim=1)
    return features.cpu().numpy()


def ensemble_with_meta_model(image_path):
    """
    Given an image path, load and preprocess the image, then get predictions from:
      - Each base model (EfficientNet, ResNet50, Swin)
      - The meta-model (ensemble)
    Returns a dictionary with each model's predicted label and confidence, plus the PIL image.
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # EfficientNet
        eff_out = efficientnet(input_tensor)
        eff_prob = torch.sigmoid(eff_out).item()
        eff_pred = "REAL" if eff_prob > 0.5 else "FAKE"

        # ResNet50
        res_out = resnet50(input_tensor)
        res_prob = torch.sigmoid(res_out).item()
        res_pred = "REAL" if res_prob > 0.5 else "FAKE"

        # Swin Transformer
        swin_out = swin(input_tensor)
        swin_prob = torch.sigmoid(swin_out).item()
        swin_pred = "REAL" if swin_prob > 0.5 else "FAKE"

    features = extract_features_batch(input_tensor)  
    meta_pred_idx = meta_model.predict(features)[0]  
    meta_prediction = class_labels[meta_pred_idx]
    meta_probs = meta_model.predict_proba(features)[0]
    meta_conf = np.max(meta_probs) * 100  

    return {
        "EfficientNet": {"prediction": eff_pred, "confidence": eff_prob * 100},
        "ResNet50": {"prediction": res_pred, "confidence": res_prob * 100},
        "Swin": {"prediction": swin_pred, "confidence": swin_prob * 100},
        "MetaModel": {"prediction": meta_prediction, "confidence": meta_conf},
        "Image": image
    }


selftest_folder = "H:/PythonProjects/ml-proj/ml-proj/es-ds/selftest"
image_files = [os.path.join(selftest_folder, f) for f in os.listdir(selftest_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

num_images = len(image_files)
cols = 5 
rows = (num_images // cols) + (1 if num_images % cols != 0 else 0)

fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
axes = axes.flatten()

for idx, image_path in enumerate(image_files):
    results = ensemble_with_meta_model(image_path)
    axes[idx].imshow(results["Image"])
    axes[idx].axis("off")
    title_text = (
        f"Eff: {results['EfficientNet']['prediction']} ({results['EfficientNet']['confidence']:.1f}%)\n"
        f"Res: {results['ResNet50']['prediction']} ({results['ResNet50']['confidence']:.1f}%)\n"
        f"Swin: {results['Swin']['prediction']} ({results['Swin']['confidence']:.1f}%)\n"
        f"Meta: {results['MetaModel']['prediction']} ({results['MetaModel']['confidence']:.1f}%)"
    )
    axes[idx].set_title(title_text, fontsize=8)

for i in range(num_images, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.show()
