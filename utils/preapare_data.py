import os
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from config.paths import TRAIN_DATASET_PATH, VAL_DATASET_PATH, LABELS_DIR, DATA_DIR

def create_dataset(image_paths, labels, transform=None):
    images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
    if transform:
        images = [transform(img) for img in images]
    tensor_imgs = torch.stack(images)
    tensor_labels = torch.tensor(labels, dtype=torch.float32)
    return TensorDataset(tensor_imgs, tensor_labels)

def get_datasets(force_recreate=False):
    # Load labels
    labels = pd.read_csv(LABELS_DIR, header=None).iloc[0].values

    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
     sorted(DATA_DIR.glob("*.jpg")), labels, test_size=0.2, random_state=42
    )

    # Load or create datasets
    if not force_recreate and os.path.exists(TRAIN_DATASET_PATH) and os.path.exists(VAL_DATASET_PATH):
        train_dataset = torch.load(TRAIN_DATASET_PATH)
        val_dataset = torch.load(VAL_DATASET_PATH)
        print('Datasets are loaded.')
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        train_dataset = create_dataset(train_paths, train_labels, transform)
        val_dataset = create_dataset(val_paths, val_labels, transform)
        torch.save(train_dataset, TRAIN_DATASET_PATH)
        torch.save(val_dataset, VAL_DATASET_PATH)
        print('Datasets are created.')

    return train_dataset, val_dataset, train_paths, val_paths
