import numpy as np
from PIL import Image
from config import IMG_HEIGHT, IMG_WIDTH

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")

    original = image.copy()

    image = image.resize((IMG_WIDTH, IMG_HEIGHT))

    image = np.array(image, dtype=np.float32)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image, original