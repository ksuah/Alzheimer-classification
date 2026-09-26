import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze()

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        cam = cam.cpu().detach().numpy()
        cam = cv2.resize(cam, (224, 224))

        return cam, class_idx


def apply_gradcam_to_folder(folder_path, model_path="best_model.pth"):
    model, device = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    target_layer = model.backbone.layer4[-1]

    gradcam = GradCAM(model, target_layer)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")

        input_tensor = transform(img).unsqueeze(0).to(device)

        cam, class_idx = gradcam.generate(input_tensor)
        pred_class = CLASSES[class_idx]

        img_np = np.array(img.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = np.uint8(0.4 * heatmap + 0.6 * img_np)

        out_path = os.path.join(folder_path, filename.split('.')[0] + "_gradcam.jpg")
        Image.fromarray(overlay).save(out_path)

        print(f"{filename} → {pred_class} → saved {out_path}")


if __name__ == "__main__":
    apply_gradcam_to_folder("test_images")
