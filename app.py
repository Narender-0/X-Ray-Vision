import streamlit as st

from preprocess import preprocess_image

from predict import predict, model

from gradcam import LayerByLayerGradCAM, overlay_heatmap

st.set_page_config(
    page_title="Chest X-Ray Diagnosis",
    layout="centered"
)

st.title("🩻 Explainable AI for Chest X-Ray Diagnosis")

uploaded = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    # Preprocess image
    image, original = preprocess_image(uploaded)

    # Display uploaded image
    st.image(
        original,
        caption="Uploaded Image",
        width=350
    )

    # Predict
    disease, confidence, prediction = predict(image)

    st.success(f"Prediction: {disease}")

    st.metric(
        label="Confidence",
        value=f"{confidence*100:.2f}%"
    )

    # Generate Grad-CAM
    gradcam = LayerByLayerGradCAM(model)

    heatmap, predicted_class = gradcam.compute_heatmap(image)

    overlay = overlay_heatmap(original, heatmap)

    st.subheader("Grad-CAM Visualization")

    st.image(
        overlay,
        caption="Model Attention Heatmap",
        use_container_width=True
    )
