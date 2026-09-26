import os
import shutil
import random

# Папка, где лежат 4 исходные папки классов
SOURCE_DIR = "data"

# Куда создадим train/val/test
TARGET_DIRS = ["train", "val", "test"]

# Пропорции
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Классы (папки)
classes = [
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented"
]

# Создаём структуру папок
for split in TARGET_DIRS:
    for cls in classes:
        os.makedirs(os.path.join(SOURCE_DIR, split, cls), exist_ok=True)

# Разделяем данные
for cls in classes:
    class_dir = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(class_dir)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # Копируем файлы
    for fname in train_files:
        shutil.copy(
            os.path.join(class_dir, fname),
            os.path.join(SOURCE_DIR, "train", cls, fname)
        )

    for fname in val_files:
        shutil.copy(
            os.path.join(class_dir, fname),
            os.path.join(SOURCE_DIR, "val", cls, fname)
        )

    for fname in test_files:
        shutil.copy(
            os.path.join(class_dir, fname),
            os.path.join(SOURCE_DIR, "test", cls, fname)
        )

print("Готово! Данные успешно разделены на train/val/test.")
