import numpy as np
from PIL import Image
from config import IMG_HEIGHT, IMG_WIDTH


def preprocess_image(uploaded_file):
    """
    Preprocess uploaded image for prediction.
    """

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Save original image for Grad-CAM visualization
    original = image.copy()

    # Resize
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))

    # Convert to NumPy
    image = np.asarray(image, dtype=np.float32)

    # Normalize
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image, original
