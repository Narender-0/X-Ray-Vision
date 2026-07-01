import tensorflow as tf
import numpy as np

from config import CLASS_NAMES


# Load model only once
model = tf.keras.models.load_model("best_model.h5")


def predict(image):
    """
    Predict disease from a preprocessed image.

    Returns:
        disease (str)
        confidence (float)
        probabilities (ndarray)
    """

    probabilities = model.predict(image, verbose=0)

    predicted_index = int(np.argmax(probabilities))

    confidence = float(probabilities[0][predicted_index])

    disease = CLASS_NAMES[predicted_index]

    return disease, confidence, probabilities
