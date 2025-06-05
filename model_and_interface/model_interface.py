import torch
from torchvision import transforms, models
from PIL import Image
import json

def predict_species(image_path, model_path="model_and_interface/05062025resnet34.pth", class_names_path="model_and_interface/class_names.json", device=None):
    # Use GPU if available
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names
    with open(class_names_path, "rb") as f:
        class_names = json.load(f)

    # Load model
    model = models.resnet34(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define transform (must match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5210, 0.5474, 0.5422], 
                             [0.1541, 0.1575, 0.1697])
    ])

    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        predicted_class = class_names[pred.item()]

    return predicted_class.replace("_", " ")

