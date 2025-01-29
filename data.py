import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

DATA_DIR = r'C:\Users\Harshan\Downloads\main-main\dataset\images\images'

def load_images_and_labels(target_size=(200, 200)):
    images, labels = [], []
    classes = sorted(os.listdir(DATA_DIR))  # Sorted for consistent labeling
    class_indices = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        class_dir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(class_dir):
            print(f"Skipping {class_dir}, not a directory.")
            continue

        # Check for 'default' and 'real_world' folders
        for subfolder in ['default', 'real_world']:
            subfolder_path = os.path.join(class_dir, subfolder)
            if not os.path.exists(subfolder_path):
                print(f"Skipping {subfolder_path}, not found.")
                continue

            for file in os.listdir(subfolder_path):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(subfolder_path, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, target_size)
                        images.append(img)
                        labels.append(class_indices[cls])
                    else:
                        print(f"Failed to load image: {img_path}")

    if not images:
        raise ValueError("No images found in the dataset directory.")
    return np.array(images), np.array(labels), classes

images, labels, classes = load_images_and_labels()
X_train, X_test, y_train, y_test = train_test_split(images / 255.0, labels, test_size=0.2, random_state=42)

print(f"Classes: {classes}")
print(f"Total Images: {len(images)}")
print(f"Training Images: {len(X_train)}, Testing Images: {len(X_test)}")
