import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

CLASS_NAMES = [
    "airplane",
    "basketball",
    "cat",
    "eye",
    "axe",
    "stairs",
    "t-shirt",
    "bicycle",
    "car",
    "stop_sign",
    "tree",
    "mug"
]


# Load data
def load_data():
    images = []
    labels = []
    
    for index, name in enumerate(CLASS_NAMES):
        data = np.load(f"Datasets/{name}.npy")[:3000] / 255.0
        label = np.full(len(data), index)
        images.append(data)
        labels.append(label)

    X = np.concatenate(images).reshape(-1, 28, 28, 1)
    y = np.concatenate(labels)

    return train_test_split(X, y, test_size=0.2, random_state=42)