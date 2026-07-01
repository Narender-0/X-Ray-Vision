#  Explainable AI for Multi-Class Chest X-Ray Diagnosis

> A deep learning-powered web application for automated multi-class chest X-ray classification with explainable AI using **Grad-CAM**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project presents an end-to-end **Explainable Artificial Intelligence (XAI)** system for automated diagnosis of chest diseases using chest X-ray images.

The application classifies uploaded images into one of the following categories:

-  Normal
-  Pneumonia
-  Tuberculosis
-  Non-Xray

Unlike conventional image classifiers, the application also provides **Grad-CAM visual explanations**, enabling users to understand which regions of the X-ray influenced the model's prediction.

The project is deployed as an interactive **Streamlit web application**, allowing users to upload their own chest X-rays or test the model using built-in sample images.

---

#  Features

-  Multi-class Chest X-ray Classification
-  Custom CNN Architecture
-  Explainable AI using Grad-CAM
-  Interactive Prediction Confidence Chart
-  Upload Custom Images
-  Built-in Sample Images
-  Prediction Confidence Scores
-  Professional Medical Dashboard
-  Real-time Inference
-  Streamlit Deployment

---

#  Project Structure

```text
Explainable_AI_for_Multi_Class_Chest_X-Ray_Diagnosis/

│
├── app.py
├── predict.py
├── preprocess.py
├── gradcam.py
├── config.py
├── requirements.txt
├── best_model.h5
├── README.md
│
├── sample_images/
│   ├── Normal/
│   ├── Pneumonia/
│   ├── Tuberculosis/
│   └── Non-Xray/
│
└── assets/
```

---

#  Model Pipeline

```text
Chest X-Ray
      │
      ▼
Image Preprocessing
      │
      ▼
Custom CNN Model
      │
      ▼
Prediction
      │
      ▼
Confidence Scores
      │
      ▼
Grad-CAM
      │
      ▼
Visual Explanation
```

---

#  Technologies Used

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras

### Computer Vision

- OpenCV
- Pillow

### Data Processing

- NumPy
- Pandas

### Visualization

- Matplotlib
- Plotly

### Explainable AI

- Grad-CAM

### Deployment

- Streamlit

---

#  Supported Classes

| Class | Description |
|--------|-------------|
| Normal | Healthy chest X-ray |
| Pneumonia | Lung infection detected |
| Tuberculosis | Tuberculosis detected |
| Non-Xray | Invalid / Non Chest X-ray Image |

---

#  Web Application Features

- Upload Chest X-ray Images
- Built-in Sample Images
- Interactive Dashboard
- Prediction Confidence Visualization
- Grad-CAM Heatmap
- Disease Information
- Responsive Medical UI

---

#  Installation

Clone the repository

```bash
git clone https://github.com/Narender-0/X-Ray-Vision.git
```

Move into the project directory

```bash
cd X-Ray-Vision
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🌐 Live Demo

> **Streamlit App**

Add your deployment link after deployment:

```
https://x-ray-vision-n.streamlit.app/
```

---

#  Screenshots

Add screenshots after deployment.

Suggested screenshots:

- Home Page
- Upload Interface
- Prediction Result
- Grad-CAM Visualization
- Confidence Chart

---

#  Explainable AI

The application integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to improve transparency.

Grad-CAM highlights the image regions that contributed most to the model's prediction, making the decision process more interpretable.

---

#  Medical Disclaimer

This application is developed for **educational and research purposes only**.

It should **not** be used as a substitute for professional medical diagnosis or treatment.

Always consult a qualified healthcare professional.

---

#  Author

**Narender Kumar & Subham Karmakar**

M.Sc. Mathematics

Indian Institute of Technology Guwahati

GitHub: https://github.com/Narender-0

---

#  If you found this project useful

Please consider giving the repository a **Star**.
