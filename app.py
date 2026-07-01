import streamlit as st

import tensorflow as tf

from preprocess import preprocess_image

from predict import predict

from predict import model

from gradcam import GradCAM

import matplotlib.pyplot as plt

st.set_page_config(page_title="Chest X-Ray Diagnosis")

st.title("Explainable AI for Chest X-Ray Diagnosis")

uploaded = st.file_uploader(
    "Upload Chest X-Ray",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image, original = preprocess_image(uploaded)

    st.image(original,width=350)

    disease, confidence, prediction = predict(image)

    st.success(f"Prediction : {disease}")

    st.write(f"Confidence : {confidence*100:.2f}%")

    gradcam = GradCAM(model)

    heatmap = gradcam.generate(image)

    fig, ax = plt.subplots()

    ax.imshow(original)

    ax.imshow(
        heatmap,
        cmap="jet",
        alpha=0.4
    )

    ax.axis("off")

    st.pyplot(fig)