import math
import os
import warnings
import cv2
warnings.filterwarnings("ignore")
from flask import Flask, request, render_template
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import torch
import numpy as np
from torchvision import models, transforms
import joblib
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# configurations
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['GRADCAM_FOLDER'] = 'static/gradcam'

# load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 1

# EfficientNet
efficientnet = models.efficientnet_b3(pretrained=False)
efficientnet.classifier[1] = torch.nn.Linear(efficientnet.classifier[1].in_features, num_classes)
efficientnet.load_state_dict(torch.load("efficientNetB3_v1.0_EXPANDED.pth", map_location=device), strict=False)
efficientnet.eval()
efficientnet.to(device)

# ResNet50
resnet50 = models.resnet50(pretrained=False)
resnet50.fc = torch.nn.Linear(resnet50.fc.in_features, num_classes)
resnet50.load_state_dict(torch.load("resNet50_v1.0_EXPANDED.pth", map_location=device))
resnet50.eval()
resnet50.to(device)

# Swin
swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
swin.head = torch.nn.Linear(swin.head.in_features, num_classes)
swin.load_state_dict(torch.load("swin_t_v1.0_EXPANDED.pth", map_location=device))
swin.eval()
swin.to(device)
# print swin model info
# for i, module in enumerate(swin.features):
#     print(f"features[{i}]: {module}")


# load meta model
meta_model = joblib.load("meta_model_rf_v1.pkl")
class_labels = ["FAKE", "REAL"]


# pre-processing for images
common_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


# gradcam generation function
def generate_gradcam(model, target_layer, input_tensor):
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    rgb_image = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
    cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    return cam_image


def reshape_transform(tensor):
    if tensor.dim() == 3:
        B, N, C = tensor.size()
        H = W = int(math.sqrt(N))
        return tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
    elif tensor.dim() == 4:
        return tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.size()}")


# gradcam generation for swin
def generate_swin_gradcam(model, input_tensor, class_idx=0):
    target_layer = model.features[7][-1].norm2

    cam = GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )

    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Use the first image

    rgb_image = input_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min() + 1e-8)
    cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    return cam_image


def generate_json_metadata(regions, output_path):
    with open(output_path, 'w') as f:
        json.dump({"regions": regions}, f, indent=4)


def identify_regions(gradcam_image_path, threshold=0.5):
    # load gradcam image
    gradcam = cv2.imread(gradcam_image_path, cv2.IMREAD_GRAYSCALE)

    # normalize to [0, 1]
    gradcam_norm = gradcam / 255.0

    # thresholding to isolate highlighted regions
    _, binary_mask = cv2.threshold(gradcam_norm, threshold, 1, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)

    # find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        regions.append({"coords": (x, y, w, h)})

    return regions


def compute_region_metrics(rgb_image_path, regions):
    rgb_image = cv2.imread(rgb_image_path)
    for region in regions:
        x, y, w, h = region["coords"]
        region_crop = rgb_image[y:y+h, x:x+w]

        # compute variance (texture complexity)
        variance = np.var(region_crop)

        # compute sharpness (Laplacian variance)
        gray_crop = cv2.cvtColor(region_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

        # save metrics
        region["variance"] = variance
        region["sharpness"] = laplacian_var
        region["laplacian_var"] = laplacian_var

    return regions


def assign_explanations(regions):
    explanations = []

    for region in regions:
        explanation = ""

        if region["variance"] < 50:
            explanation += "This region has unnatural smoothness typical in AI-generated imagery. "

        if region["laplacian_var"] < 20:
            explanation += "Lack of sharpness here indicates potential synthetic textures. "

        if not explanation:
            explanation = "Region appears natural but was influential due to subtle anomalies."

        explanations.append({
            "coords": region["coords"],
            "explanation": explanation.strip()
        })

    return explanations


def grad_rollout_call(grad_rollout_obj, input_tensor, category_index):
    return grad_rollout_obj(input_tensor, category_index)


# extract features function
def extract_features_batch(batch_images):
    batch_images = batch_images.to(device)

    with torch.no_grad():
        eff_out = efficientnet(batch_images)
    eff_prob = torch.sigmoid(eff_out)

    with torch.no_grad():
        res_out = resnet50(batch_images)
    res_prob = torch.sigmoid(res_out)

    with torch.no_grad():
        swin_out = swin(batch_images)
    swin_prob = torch.sigmoid(swin_out)

    features = torch.cat([eff_prob, res_prob, swin_prob], dim=1)
    return features.cpu().numpy()


# ensemble prediction function
def ensemble_with_meta_model(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = common_transform(image).unsqueeze(0).to(device)

    # individual model preds
    with torch.no_grad():
        eff_out = efficientnet(input_tensor)
        eff_prob = torch.sigmoid(eff_out).item()
        eff_pred = "REAL" if eff_prob > 0.5 else "FAKE"

        res_out = resnet50(input_tensor)
        res_prob = torch.sigmoid(res_out).item()
        res_pred = "REAL" if res_prob > 0.5 else "FAKE"

        swin_out = swin(input_tensor)
        swin_prob = torch.sigmoid(swin_out).item()
        swin_pred = "REAL" if swin_prob > 0.5 else "FAKE"

    def compute_pred(prob):
        if prob > 0.5:
            return "REAL", prob * 100
        else:
            return "FAKE", (1 - prob) * 100

    eff_pred, eff_conf = compute_pred(eff_prob)
    res_pred, res_conf = compute_pred(res_prob)
    swin_pred, swin_conf = compute_pred(swin_prob)

    # meta model pred
    with torch.no_grad():
        eff_prob_tensor = torch.sigmoid(efficientnet(input_tensor))
        res_prob_tensor = torch.sigmoid(resnet50(input_tensor))
        swin_prob_tensor = torch.sigmoid(swin(input_tensor))
        features = torch.cat([eff_prob_tensor, res_prob_tensor, swin_prob_tensor], dim=1)
        features_np = features.cpu().numpy()
    meta_pred_idx = meta_model.predict(features_np)[0]
    meta_prediction = class_labels[meta_pred_idx]
    meta_probs = meta_model.predict_proba(features_np)[0]
    meta_conf = np.max(meta_probs) * 100

    # resnet gradcam
    res_layer = resnet50.layer4[-1]
    res_cam_np = generate_gradcam(resnet50, res_layer, input_tensor)
    res_cam_file = os.path.join(app.config["GRADCAM_FOLDER"], "res_cam_" + os.path.basename(image_path))
    res_cam_name = "res_cam_" + os.path.basename(image_path)
    Image.fromarray(res_cam_np).save(res_cam_file)
    print(res_cam_file)

    # efficientnet gradcam
    eff_layer = efficientnet.features[-1][0]
    eff_cam_np = generate_gradcam(efficientnet, eff_layer, input_tensor)
    eff_cam_file = os.path.join(app.config["GRADCAM_FOLDER"], "eff_cam_" + os.path.basename(image_path))
    eff_cam_name = "eff_cam_" + os.path.basename(image_path)
    Image.fromarray(eff_cam_np).save(eff_cam_file)

    # Swin gradcam
    swin_cam_np = generate_swin_gradcam(swin, input_tensor)
    swin_cam_file = os.path.join(app.config["GRADCAM_FOLDER"], "swin_cam_" + os.path.basename(image_path))
    swin_cam_name = "swin_cam_" + os.path.basename(image_path)
    Image.fromarray(swin_cam_np).save(swin_cam_file)

    eff_cam_basename = os.path.splitext(os.path.basename(eff_cam_file))[0]
    gray_cam_filename = eff_cam_basename + "_gray.png"
    gray_cam_path = os.path.join(os.path.dirname(eff_cam_file), gray_cam_filename)

    cv2.imwrite(gray_cam_path, cv2.cvtColor(eff_cam_np, cv2.COLOR_RGB2GRAY))

    # identify regions for XAI
    regions = identify_regions(gray_cam_path, threshold=0.5)

    # compute metrics for explanation
    regions = compute_region_metrics(image_path, regions)

    # assign explanations
    explained_regions = assign_explanations(regions)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # save explanations as json
    explanation_json_path = os.path.join(app.config["GRADCAM_FOLDER"], f"explanations_{base_filename}.json")
    print("Saving explanations JSON to:", explanation_json_path)
    generate_json_metadata(explained_regions, explanation_json_path)

    return {
        "EfficientNet": {"prediction": eff_pred, "confidence": eff_conf},
        "ResNet50": {"prediction": res_pred, "confidence": res_conf},
        "Swin": {"prediction": swin_pred, "confidence": swin_conf},
        "MetaModel": {"prediction": meta_prediction, "confidence": meta_conf},
        "Image": image,
        "ResCAM": res_cam_name,
        "EffCAM": eff_cam_name,
        "SwinCAM": swin_cam_name,
        "explanations": os.path.basename(explanation_json_path)
    }


# flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    eff_pred = res_pred = swin_pred = meta_prediction = None
    eff_conf = res_conf = swin_conf = meta_conf = None
    res_cam_file = eff_cam_file = swin_cam_file = None
    explanations_file = None

    if request.method == "POST":
        file = request.files.get("image_file")
        if file and file.filename:
            filename = file.filename
            saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(saved_path)

            results = ensemble_with_meta_model(saved_path)
            eff_pred = results["EfficientNet"]["prediction"]
            eff_conf = results["EfficientNet"]["confidence"]
            res_pred = results["ResNet50"]["prediction"]
            res_conf = results["ResNet50"]["confidence"]
            swin_pred = results["Swin"]["prediction"]
            swin_conf = results["Swin"]["confidence"]
            meta_prediction = results["MetaModel"]["prediction"]
            meta_conf = results["MetaModel"]["confidence"]
            res_cam_file = results["ResCAM"]
            eff_cam_file = results["EffCAM"]
            swin_cam_file = results["SwinCAM"]
            explanations_file = results.get("explanations", "")
            saved_path = filename
        else:
            meta_prediction = "No image file uploaded."
            explanations_file = ""
    else:
        saved_path = ""

    return render_template(
        "index.html",
        eff_pred=eff_pred,
        eff_conf=eff_conf,
        res_pred=res_pred,
        res_conf=res_conf,
        swin_pred=swin_pred,
        swin_conf=swin_conf,
        meta_prediction=meta_prediction,
        meta_conf=meta_conf,
        res_cam=res_cam_file,
        eff_cam=eff_cam_file,
        swin_cam=swin_cam_file,
        explanations=explanations_file,
        original_image=saved_path
    )


@app.route("/about")
def about():
    return ""


if __name__ == "__main__":
    app.run(debug=True)
