import os
import torch
from torchvision import transforms
from PIL import Image
from model import create_model

CLASSES = [
    "Non_Demented",
    "Very_Mild_Demented",
    "Mild_Demented",
    "Moderate_Demented"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_single_image(model, device, image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)

    class_name = CLASSES[predicted.item()]
    return class_name


def predict_folder(folder_path, model_path="best_model.pth"):
    model, device = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\nПапка: {folder_path}\n")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            pred = predict_single_image(model, device, img_path)
            print(f"{filename} → {pred}")


if __name__ == "__main__":
    predict_folder("test_images")
