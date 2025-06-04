import os 
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
DATA_DIR = "F:/birdData/birdDataset"
BATCH_SIZE = 32
EPOCHS = 10 #number of loops through the dataset
LR = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 2
print("CUDA available: ", torch.cuda.is_available())
print("Device name:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#use an unnormalised transform to get the values we should use to normalise the pixel values
"""
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("Custom Mean:", mean)
    print("Custom Std: ", std)
Custom Mean: tensor([0.5210, 0.5474, 0.5422])
Custom Std:  tensor([0.1541, 0.1575, 0.1697])
"""

def main():
    best_accuracy = 0
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5210, 0.5474, 0.5422], 
                             [0.1541, 0.1575, 0.1697])  # the custom values that I got earlier
    ])
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"{DATA_DIR} does not exist.")
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    class_names = dataset.classes
   

    val_split = 0.2
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    #train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    losses = []
    accuracies = []
    for epoch in range(EPOCHS): #this is the training loop
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())
        losses.append(total_loss)
        model.eval()
        with torch.no_grad():
            val_correct = 0
            val_total = 0
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
            val_accuracy = val_correct / val_total
            print(f"Validation Accuracy: {val_accuracy:.4f}")
        accuracies.append(correct / total)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {correct/total:.4f}")
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), accuracies, label="Accuracy", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()



    torch.save(model.state_dict(), "final_model.pth")



if __name__ == "__main__":
    main()