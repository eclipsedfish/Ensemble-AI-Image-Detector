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
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])


resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 1)
# Load the checkpoint; update the path if needed.
checkpoint_path = "resNet50_v1.0_EXPANDED.pth"
resnet.load_state_dict(torch.load(checkpoint_path, map_location=device))
resnet = resnet.to(device)
resnet.eval()
print("ResNet model loaded successfully.")


def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device) 

    with torch.no_grad():
        output = resnet(input_tensor).squeeze() 
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

fig.suptitle("ResNet50 EXPANDED Test", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
