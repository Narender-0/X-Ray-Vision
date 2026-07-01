import tensorflow as tf
import numpy as np
import cv2


class LayerByLayerGradCAM:
    """
    Robust Grad-CAM implementation for custom CNN models.
    """

    def __init__(self, model):
        self.model = model

        # Build model if not already built
        sample = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
        _ = self.model(sample, training=False)

        self.target_layer = self._find_best_layer()

    def _find_best_layer(self):
        """
        Find the last Conv2D layer.
        """

        conv_layers = []

        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append(layer)

        if len(conv_layers) > 0:
            return conv_layers[-1]

        raise ValueError("No Conv2D layer found in the model.")

    def compute_heatmap(self, image, class_idx=None):
        """
        Generate Grad-CAM heatmap.

        Parameters
        ----------
        image : numpy array
            Shape = (1,224,224,3)

        Returns
        -------
        heatmap
        predicted_class
        """

        if isinstance(image, np.ndarray):
            image = tf.convert_to_tensor(image, dtype=tf.float32)

        target_output = None

        with tf.GradientTape() as tape:

            x = image

            for layer in self.model.layers:

                x = layer(x, training=False)

                if layer == self.target_layer:
                    target_output = x
                    tape.watch(target_output)

            predictions = x

            if class_idx is None:
                class_idx = tf.argmax(predictions[0])

            score = predictions[:, class_idx]

        grads = tape.gradient(score, target_output)

        if grads is None:

            heatmap = np.zeros((224, 224))

            return heatmap, int(class_idx.numpy())

        weights = tf.reduce_mean(grads, axis=(1, 2))

        heatmap = tf.reduce_sum(
            target_output *
            weights[:, tf.newaxis, tf.newaxis, :],
            axis=-1
        )

        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)

        if tf.reduce_max(heatmap) > 0:
            heatmap /= tf.reduce_max(heatmap)

        heatmap = heatmap.numpy()

        heatmap = cv2.resize(
            heatmap,
            (224, 224)
        )

        return heatmap, int(class_idx.numpy())


def overlay_heatmap(original_image, heatmap):
    """
    Overlay Grad-CAM heatmap on original image.

    Parameters
    ----------
    original_image : PIL Image
    heatmap : ndarray

    Returns
    -------
    overlay image
    """

    image = np.array(original_image)

    image = cv2.resize(image, (224, 224))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image,
        0.6,
        heatmap,
        0.4,
        0
    )

    return overlay
