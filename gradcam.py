import tensorflow as tf
import numpy as np
import cv2

class GradCAM:

    def __init__(self, model):

        self.model = model

        self.last_conv = None

        for layer in reversed(model.layers):

            if isinstance(layer, tf.keras.layers.Conv2D):

                self.last_conv = layer.name

                break

    def generate(self, image):

        grad_model = tf.keras.models.Model(
            self.model.inputs,
            [self.model.get_layer(self.last_conv).output,
             self.model.output]
        )

        with tf.GradientTape() as tape:

            conv_output, predictions = grad_model(image)

            class_idx = tf.argmax(predictions[0])

            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_output)

        pooled = tf.reduce_mean(grads, axis=(0,1,2))

        conv_output = conv_output[0]

        heatmap = conv_output @ pooled[..., tf.newaxis]

        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap,0)

        heatmap /= tf.reduce_max(heatmap)

        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap,(224,224))

        return heatmap