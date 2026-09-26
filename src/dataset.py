import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class AlzheimerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.class_to_idx = {
            "Non_Demented": 0,
            "Very_Mild_Demented": 1,
            "Mild_Demented": 2,
            "Moderate_Demented": 3
        }
        print("CLASS ORDER:", self.class_to_idx)

        self.image_paths = []
        self.labels = []

        for class_name, class_idx in self.class_to_idx.items():
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue

            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                self.image_paths.append(img_path)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = AlzheimerDataset(os.path.join(data_dir, "train"), transform)
    val_dataset = AlzheimerDataset(os.path.join(data_dir, "val"), transform)
    test_dataset = AlzheimerDataset(os.path.join(data_dir, "test"), transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
