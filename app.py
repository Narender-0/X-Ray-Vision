import streamlit as st
import plotly.express as px
import pandas as pd
import os
from preprocess import preprocess_image
from predict import predict, model
from gradcam import LayerByLayerGradCAM, overlay_heatmap

from config import CLASS_NAMES

# ---------------------- PAGE CONFIG ----------------------

st.set_page_config(
    page_title="AI Chest X-Ray Diagnosis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- CUSTOM CSS ----------------------

st.markdown("""
<style>

.main{
    background-color:#F8F9FA;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    padding-left:3rem;
    padding-right:3rem;
}

h1{
    color:#0B5394;
}

div[data-testid="stMetric"]{
    background-color:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 0px 8px rgba(0,0,0,0.1);
}

.stButton>button{
    background-color:#0B5394;
    color:white;
    border-radius:10px;
    height:50px;
    font-size:18px;
}

.stButton>button:hover{
    background-color:#1976D2;
    color:white;
}

.upload-box{
    border:2px dashed #1976D2;
    border-radius:15px;
    padding:20px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)


def get_sample_images(folder):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

SAMPLE_IMAGES = {
    "Normal": get_sample_images("sample_images/Normal"),
    "Pneumonia": get_sample_images("sample_images/Pneumonia"),
    "Tuberculosis": get_sample_images("sample_images/Tuberculosis"),
    "Non-Xray": get_sample_images("sample_images/Nonxray")
}

# ---------------------- SIDEBAR ----------------------

with st.sidebar:

    st.title("🩻 Chest X-Ray AI")
    st.subheader("Try Sample Images")

    selected_category = st.selectbox(

    "Disease Category",

    list(SAMPLE_IMAGES.keys()))

    selected_image = st.selectbox(

    "Choose Image",

    SAMPLE_IMAGES[selected_category])

    use_sample = st.button("Load Sample Image")

    st.markdown("---")

   

    st.subheader("Model")

    st.write("Custom CNN")

    st.write("TensorFlow")

    st.write("Grad-CAM Explainability")

    st.markdown("---")
    
    st.subheader("Developer")
    
    st.write("Narender Kumar")

    st.write("Subham Karmakar")

    st.write("M.Sc Mathematics")

    st.write("IIT Guwahati")

# ---------------------- TITLE ----------------------

st.markdown("""
# 🩻 AI Chest X-Ray Disease Diagnosis

### Deep Learning based Explainable AI System

Upload a Chest X-Ray image to automatically detect:

- Normal
- Pneumonia
- Tuberculosis
- Non-Xray

The prediction is accompanied by **Grad-CAM** visualization showing
which regions of the image influenced the model's decision.

---
""")

# ---------------------- DISEASE INFORMATION ----------------------

DISEASE_INFO = {

    "Normal":
    """
     **No abnormality detected**

    The uploaded X-ray appears normal.

    Continue regular health checkups.

    Clinical confirmation is still recommended.
    """,

    "Pneumonia":
    """
     **Pneumonia Detected**

    Pneumonia is an infection that inflames the air sacs of one or both lungs.

    Common symptoms:

    • Fever

    • Cough

    • Chest pain

    • Difficulty breathing

    Please consult a healthcare professional.
    """,

    "Tuberculosis":
    """
     **Tuberculosis Detected**

    Tuberculosis is a bacterial disease affecting the lungs.

    Common symptoms:

    • Persistent cough

    • Night sweats

    • Weight loss

    • Fever

    Seek immediate medical evaluation.
    """,

    "Non-Xray":
    """
    ⚠ Invalid Image

    The uploaded image does not appear to be a Chest X-Ray.

    Please upload a valid Chest X-Ray image.
    """
}

# ---------------------- FILE UPLOADER ----------------------

st.markdown("""
## Upload Your Own Chest X-Ray

**OR**

### Use the Sample Images available in the sidebar.
""")

uploaded = st.file_uploader(

    " Upload Chest X-Ray",

    type=["jpg","jpeg","png"]

)
uploaded_image = uploaded if uploaded is not None else None

if use_sample:
    uploaded_image = selected_image
    
st.markdown("---")
# ---------------------- PREDICTION ----------------------

if uploaded_image is not None:

    with st.spinner(" AI is analyzing the Chest X-Ray..."):

        # Preprocess image
        image, original = preprocess_image(uploaded_image)

        # Predict
        disease, confidence, probabilities = predict(image)

        if use_sample:
            st.info(
                f"Using sample image: {os.path.basename(selected_image)}"
        )

        # Grad-CAM
        gradcam = LayerByLayerGradCAM(model)

        heatmap, pred_class = gradcam.compute_heatmap(image)

        overlay = overlay_heatmap(original, heatmap)

    # ---------------------- LAYOUT ----------------------

    left_col, right_col = st.columns([1, 1])

    with left_col:

        st.subheader("📷 Uploaded X-Ray")

        st.image(
            original,
            width="stretch"
        )

    with right_col:

        st.subheader(" Grad-CAM Visualization")

        st.image(
            overlay,
            width="stretch"
        )

    st.markdown("---")

    # ---------------------- METRICS ----------------------

    metric1, metric2 = st.columns(2)

    with metric1:

        if disease == "Normal":

            st.success(f"###  {disease}")

        elif disease == "Pneumonia":

            st.error(f"###  {disease}")

        elif disease == "Tuberculosis":

            st.warning(f"###  {disease}")

        else:

            st.info(f"###  {disease}")

    with metric2:

        st.metric(

            label="Prediction Confidence",

            value=f"{confidence*100:.2f}%"

        )

    st.markdown("---")

    # ---------------------- PROBABILITY CHART ----------------------

    st.subheader(" Prediction Confidence for Each Class")

    probability_df = pd.DataFrame(
        {
            "Disease": CLASS_NAMES,
            "Confidence": probabilities[0] * 100
        }
    )

    fig = px.bar(
        probability_df,
        x="Confidence",
        y="Disease",
        orientation="h",
        text="Confidence",
        color="Confidence",
        color_continuous_scale="Blues"
    )

    fig.update_traces(

    texttemplate="%{text:.2f}%",

    textposition="outside",

    textfont=dict(
        size=15,
        color="black")

    )

    fig.update_layout(

    height=350,

    xaxis_title="Confidence (%)",

    yaxis_title="",

    coloraxis_showscale=False,

    plot_bgcolor="white",

    paper_bgcolor="white",

    margin=dict(l=20, r=20, t=30, b=20),

    font=dict(
        family="Arial",
        size=15,
        color="black"      # <-- Makes all text black
    ),

    xaxis=dict(
        tickfont=dict(
            size=14,
            color="black"
        ),
        title_font=dict(
            size=16,
            color="black"
        ),
        showgrid=True,
        gridcolor="lightgray"
    ),

    yaxis=dict(
        tickfont=dict(
            size=15,
            color="black"
        ),
        title_font=dict(
            color="black"
        ))

    )
    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.markdown("---")

    # ---------------------- DISEASE INFORMATION ----------------------

    st.subheader("ℹ Disease Information")

    st.info(
        DISEASE_INFO[disease]
    )

    st.markdown("---")

    # ---------------------- MODEL DETAILS ----------------------

    with st.expander(" Model Details"):

        st.markdown("""
### Model

- **Architecture:** Custom CNN
- **Framework:** TensorFlow
- **Input Size:** 224 × 224
- **Explainability:** Grad-CAM
- **Output Classes:** 4
- **Deployment:** Streamlit
        """)

    # ---------------------- DISCLAIMER ----------------------

    st.warning(
        """
⚠ **Medical Disclaimer**

This AI model is intended **only for educational and research purposes**.

It should **NOT** be used as a substitute for professional medical diagnosis.

Always consult a qualified healthcare professional before making any medical decisions.
        """
    )


# ---------------------- FOOTER ----------------------

st.markdown("---")

st.markdown(
"""
<div style='text-align:center'>

Made by using <b>TensorFlow</b>, <b>Streamlit</b> and <b>Grad-CAM</b>

<br>

<b>Narender Kumar & Subham karmakar</b>

<br>

M.Sc. Mathematics

<br>

IIT Guwahati

</div>
""",
unsafe_allow_html=True
)
