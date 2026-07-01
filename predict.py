import tensorflow as tf
import numpy as np
from config import CLASS_NAMES

model = tf.keras.models.load_model("best_model.h5")

def predict(image):

    prediction = model.predict(image, verbose=0)

    index = np.argmax(prediction)

    confidence = prediction[0][index]

    return CLASS_NAMES[index], confidence, prediction