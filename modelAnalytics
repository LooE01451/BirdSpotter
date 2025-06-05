import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json

MODEL_PATH = "D:/models/05062025resnet34/best_model.pth"
DATA_DIR = "D:/birdData/birdDataset"

with open("class_names.json", "rb") as f:
    class_names = json.load(f)


IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5210, 0.5474, 0.5422], 
                         [0.1541, 0.1575, 0.1697])
])
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
val_split = 0.2
val_size = int(val_split * len(dataset))
_, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-val_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# === Load the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

def main():

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # === Generate and plot confusion matrix
    unique, counts = np.unique(all_labels, return_counts=True)
    top_classes = np.argsort(counts)[-20:]

    # Create a mapping from original class index to new index (0-19)
    class_idx_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(top_classes)}

    # Filter and remap predictions and labels (keep only pairs where both are in top_classes)
    filtered_pairs = [
        (class_idx_map[t], class_idx_map[p])
        for p, t in zip(all_preds, all_labels)
        if t in top_classes and p in top_classes
    ]
    filtered_labels = [t for t, p in filtered_pairs]
    filtered_preds = [p for t, p in filtered_pairs]

    cm = confusion_matrix(filtered_labels, filtered_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=[class_names[i] for i in top_classes])

    fig, ax = plt.subplots(figsize=(40, 40))
    disp.plot(ax=ax, xticks_rotation=90, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()