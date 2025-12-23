import os
import shutil
import random

SOURCE_DIR = "data/PlantVillage"
TARGET_DIR = "data/raw"

CLASSES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold"
]

SPLIT_RATIO = 0.8

for cls in CLASSES:
    images = os.listdir(os.path.join(SOURCE_DIR, cls))
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_idx]
    test_images = images[split_idx:]

    os.makedirs(os.path.join(TARGET_DIR, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, "test", cls), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(SOURCE_DIR, cls, img),
            os.path.join(TARGET_DIR, "train", cls, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(SOURCE_DIR, cls, img),
            os.path.join(TARGET_DIR, "test", cls, img)
        )

print("Dataset preparation completed.")

