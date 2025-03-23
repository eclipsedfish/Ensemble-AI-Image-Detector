from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_train_dataset = datasets.ImageFolder(root="H:/PythonProjects/ml-proj/ml-proj/es-ds/cifake/train", transform=transform)
test_dataset = datasets.ImageFolder(root='H:/PythonProjects/ml-proj/ml-proj/es-ds/cifake/test', transform=transform)

train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {len(test_dataset)}")


resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, 1)  
resnet = resnet.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(resnet.parameters(), lr=1e-4)

print("Class labels:", full_train_dataset.classes)
print("Class to index mapping:", full_train_dataset.class_to_idx)


# training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}\n" + "-" * 30)

        # training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            optimizer.zero_grad()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_train / total_train * 100
        print(f"Training Loss: {train_loss:.4f} - Training Accuracy: {train_accuracy:.2f}%")

        # validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                predictions = (torch.sigmoid(outputs) >= 0.5).float()
                correct_val += (predictions == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_val / total_val * 100
        print(f"Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}%")

    return model

# testing function for final evaluation
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs).squeeze()

            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    all_preds_binary = (np.array(all_preds) >= 0.5).astype(int)

    # metrics
    test_accuracy = accuracy_score(all_labels, all_preds_binary)
    print(f"\nTest Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds_binary))
    print(f"F1-Score: {f1_score(all_labels, all_preds_binary):.4f}")
    print(f"Recall: {recall_score(all_labels, all_preds_binary):.4f}")
    print(f"Precision: {precision_score(all_labels, all_preds_binary):.4f}")

# train and evaluate
start_time = datetime.now()
resnet = train_model(resnet, train_loader, val_loader, criterion, optimizer, num_epochs=10)
end_time = datetime.now()

print(f"\nSTART TIME: {start_time}")
print(f"END TIME: {end_time}")
duration = (end_time - start_time).total_seconds()
print(f"DURATION: {timedelta(seconds=duration)}")

# evaluate on test set
evaluate_model(resnet, test_loader, criterion)

# save
torch.save(resnet.to('cpu').state_dict(), "resNet50_v1.0_CIFAKE.pth")
print("\nResNet model saved as resNet50_v1.0_CIFAKE.pth")
