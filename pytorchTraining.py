import os 
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
DATA_DIR = "D:/birdData/birdDataset"
BATCH_SIZE = 32
EPOCHS = 10 #number of loops through the dataset
LR = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 2
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
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

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
        for imgs, labels in train_loader:
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
        losses.append(total_loss)
        accuracies.append(correct / total)

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

     


    # model is saved
    model.eval()
    torch.save(model.state_dict(), "bird_species_classifier.pth")
    torch.save(model, "bird_species_classifier_full.pth")
    print("✅ Model saved to bird_species_classifier.pth")

if __name__ == "__main__":
    main()