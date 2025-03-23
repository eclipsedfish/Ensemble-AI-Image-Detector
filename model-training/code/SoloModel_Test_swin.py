import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
swin.head = nn.Linear(swin.head.in_features, 1)
swin = swin.to(device)
swin.eval()

checkpoint_path = "swin_t_v1.0_EXPANDED.pth"  
try:
    state_dict = torch.load(checkpoint_path, map_location=device)
    swin.load_state_dict(state_dict, strict=True)
except RuntimeError as e:
    print("Error loading checkpoint strictly:", e)
    model_dict = swin.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    print(f"Loading {len(pretrained_dict)} out of {len(model_dict)} parameters.")
    model_dict.update(pretrained_dict)
    swin.load_state_dict(model_dict)
print("Swin Transformer model loaded successfully.")

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  

    with torch.no_grad():
        output = swin(input_tensor).squeeze()  
        prob = torch.sigmoid(output).item()     
    pred = "REAL" if prob > 0.5 else "FAKE"
    return pred, prob, image

selftest_folder = "H:/PythonProjects/ml-proj/ml-proj/es-ds/selftest"  
image_files = [os.path.join(selftest_folder, f) for f in os.listdir(selftest_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

num_images = len(image_files)
cols = 5
rows = (num_images // cols) + (1 if num_images % cols != 0 else 0)

fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
axes = axes.flatten()

for idx, image_path in enumerate(image_files):
    pred, prob, img = predict_image(image_path)
    axes[idx].imshow(img)
    axes[idx].axis("off")
    axes[idx].set_title(f"{pred}\n{prob * 100:.1f}%", fontsize=10)

for i in range(num_images, len(axes)):
    fig.delaxes(axes[i])

fig.suptitle("Swin Transformer EXPANDED", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
