import torch
import torch.nn as nn
from torchvision import models


class AlzheimerNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerNet, self).__init__()

        # Берём предобученную ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def create_model(device=None, num_classes=4):
    model = AlzheimerNet(num_classes=num_classes)

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available()
                              else "cpu")

    model = model.to(device)
    return model, device
