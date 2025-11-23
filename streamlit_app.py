import os
import time

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms

# =====================================================
# 1. CLASS NAMES (38) – MATCHING YOUR KAGGLE DATASET
# =====================================================
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(CLASS_NAMES)
CSV_PATH = "prediction_log.csv"

# =====================================================
# 2. STATIC CLASSIFICATION REPORT (YOUR NUMBERS)
# =====================================================
# precision, recall, f1, support for each class
STATIC_REPORT = {
    "Apple___Apple_scab":                           {"precision": 0.90, "recall": 0.99, "f1": 0.94, "support": 504},
    "Apple___Black_rot":                            {"precision": 0.99, "recall": 0.98, "f1": 0.99, "support": 497},
    "Apple___Cedar_apple_rust":                     {"precision": 1.00, "recall": 0.92, "f1": 0.96, "support": 440},
    "Apple___healthy":                              {"precision": 0.99, "recall": 0.97, "f1": 0.98, "support": 502},
    "Blueberry___healthy":                          {"precision": 0.94, "recall": 0.98, "f1": 0.96, "support": 454},
    "Cherry_(including_sour)___Powdery_mildew":     {"precision": 0.98, "recall": 0.98, "f1": 0.98, "support": 421},
    "Cherry_(including_sour)___healthy":            {"precision": 0.99, "recall": 1.00, "f1": 0.99, "support": 456},
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
                                                    {"precision": 0.97, "recall": 0.76, "f1": 0.85, "support": 410},
    "Corn_(maize)___Common_rust_":                  {"precision": 0.99, "recall": 0.99, "f1": 0.99, "support": 477},
    "Corn_(maize)___Northern_Leaf_Blight":          {"precision": 0.83, "recall": 0.97, "f1": 0.89, "support": 477},
    "Corn_(maize)___healthy":                       {"precision": 0.99, "recall": 1.00, "f1": 0.99, "support": 465},
    "Grape___Black_rot":                            {"precision": 0.96, "recall": 0.99, "f1": 0.98, "support": 472},
    "Grape___Esca_(Black_Measles)":                 {"precision": 0.99, "recall": 0.98, "f1": 0.99, "support": 480},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":   {"precision": 0.99, "recall": 0.98, "f1": 0.99, "support": 430},
    "Grape___healthy":                              {"precision": 0.99, "recall": 1.00, "f1": 0.99, "support": 423},
    "Orange___Haunglongbing_(Citrus_greening)":     {"precision": 0.99, "recall": 1.00, "f1": 1.00, "support": 503},
    "Peach___Bacterial_spot":                       {"precision": 0.95, "recall": 0.98, "f1": 0.97, "support": 459},
    "Peach___healthy":                              {"precision": 0.97, "recall": 0.98, "f1": 0.97, "support": 432},
    "Pepper,_bell___Bacterial_spot":                {"precision": 0.98, "recall": 0.99, "f1": 0.98, "support": 478},
    "Pepper,_bell___healthy":                       {"precision": 0.98, "recall": 0.89, "f1": 0.93, "support": 497},
    "Potato___Early_blight":                        {"precision": 0.99, "recall": 0.98, "f1": 0.98, "support": 485},
    "Potato___Late_blight":                         {"precision": 0.92, "recall": 0.97, "f1": 0.94, "support": 485},
    "Potato___healthy":                             {"precision": 0.98, "recall": 0.93, "f1": 0.95, "support": 456},
    "Raspberry___healthy":                          {"precision": 0.99, "recall": 1.00, "f1": 0.99, "support": 445},
    "Soybean___healthy":                            {"precision": 0.96, "recall": 0.97, "f1": 0.97, "support": 505},
    "Squash___Powdery_mildew":                      {"precision": 1.00, "recall": 1.00, "f1": 1.00, "support": 434},
    "Strawberry___Leaf_scorch":                     {"precision": 0.97, "recall": 0.99, "f1": 0.98, "support": 444},
    "Strawberry___healthy":                         {"precision": 1.00, "recall": 0.99, "f1": 0.99, "support": 456},
    "Tomato___Bacterial_spot":                      {"precision": 0.97, "recall": 0.88, "f1": 0.92, "support": 425},
    "Tomato___Early_blight":                        {"precision": 0.86, "recall": 0.85, "f1": 0.85, "support": 480},
    "Tomato___Late_blight":                         {"precision": 0.83, "recall": 0.89, "f1": 0.86, "support": 463},
    "Tomato___Leaf_Mold":                           {"precision": 0.95, "recall": 0.91, "f1": 0.93, "support": 470},
    "Tomato___Septoria_leaf_spot":                  {"precision": 0.80, "recall": 0.92, "f1": 0.86, "support": 436},
    "Tomato___Spider_mites Two-spotted_spider_mite":{"precision": 0.89, "recall": 0.92, "f1": 0.90, "support": 435},
    "Tomato___Target_Spot":                         {"precision": 0.89, "recall": 0.80, "f1": 0.84, "support": 457},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":       {"precision": 0.97, "recall": 0.98, "f1": 0.97, "support": 490},
    "Tomato___Tomato_mosaic_virus":                 {"precision": 0.95, "recall": 0.87, "f1": 0.91, "support": 448},
    "Tomato___healthy":                             {"precision": 0.97, "recall": 0.97, "f1": 0.97, "support": 481},
}

STATIC_SUMMARY = {
    "accuracy": 0.95,
    "macro_precision": 0.95,
    "macro_recall": 0.95,
    "macro_f1": 0.95,
    "weighted_precision": 0.95,
    "weighted_recall": 0.95,
    "weighted_f1": 0.95,
    "total_samples": 17572,
}

# =====================================================
# 3. MODEL LOADING – EXACT SAME AS KAGGLE TRAINING
# =====================================================
@st.cache_resource
def load_model():
    """
    Rebuild MobileNetV2 exactly as used during training:
        model = models.mobilenet_v2(pretrained=True)
        for p in model.features.parameters(): p.requires_grad = False
        model.classifier[1] = nn.Sequential(Dropout(0.2), Linear(in_features, 38))
    """
    model = models.mobilenet_v2(weights=None)

    # Freeze feature extractor
    for p in model.features.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, NUM_CLASSES),
    )

    # Load trained weights (must be in same folder)
    state_dict = torch.load("mobilenetv2_plant.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model


model = load_model()

# =====================================================
# 4. PREPROCESS – SAME AS TRAINING
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess(img: Image.Image) -> torch.Tensor:
    return transform(img).unsqueeze(0)  # [1, 3, 224, 224]


# =====================================================
# 5. EXPLAINABILITY HELPERS
# =====================================================
def health_status(cls: str) -> str:
    return "Healthy" if "healthy" in cls.lower() else "Diseased"


def severity(conf: float) -> str:
    if conf < 0.50:
        return "Uncertain"
    if conf < 0.70:
        return "Mild"
    if conf < 0.85:
        return "Moderate"
    return "Severe"


def recommendation(cls: str) -> str:
    c = cls.lower()
    if "healthy" in c:
        return "Leaf is healthy. Maintain good care."
    if "blight" in c:
        return "Remove infected leaves. Avoid wet foliage."
    if "rust" in c:
        return "Increase airflow. Keep leaves dry."
    if "spot" in c:
        return "Avoid overhead watering. Improve drainage."
    return "General plant stress. Improve airflow and remove infected parts."


# =====================================================
# 6. CSV HELPERS (PREDICTION LOG ONLY)
# =====================================================
LOG_COLUMNS = [
    "timestamp",
    "image_name",
    "predicted_class",
    "confidence",
    "health_status",
    "severity",
    "recommendation",
]


def load_csv(csv_path: str = CSV_PATH) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for col in LOG_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[LOG_COLUMNS]
    else:
        return pd.DataFrame(columns=LOG_COLUMNS)


def append_to_csv(record: dict, csv_path: str = CSV_PATH) -> pd.DataFrame:
    df_old = load_csv(csv_path)
    df_new = pd.DataFrame([record])
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all.to_csv(csv_path, index=False)
    return df_all


# =====================================================
# 7. STREAMLIT UI – PREDICTION + LOGGING
# =====================================================
st.set_page_config(page_title="Plant Disease Detector", layout="wide")

st.title("🌿 Plant Disease Detection Dashboard")
st.write("Upload a plant leaf image to detect disease and log results for analysis.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    # ---- Model prediction ----
    x = preprocess(img)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    idx = int(torch.argmax(probs).item())
    cls = CLASS_NAMES[idx]
    conf = float(probs[idx])

    h_status = health_status(cls)
    sev = severity(conf)
    rec = recommendation(cls)

    with col2:
        st.subheader("Prediction Results")
        st.write(f"**Predicted Disease:** `{cls}`")
        st.write(f"**Confidence:** `{conf:.4f}`")
        st.write(f"**Health Status:** `{h_status}`")
        st.write(f"**Severity Level:** `{sev}`")
        st.write(f"**Recommendation:** {rec}")

    # ---- Build record and log to CSV ----
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": uploaded_file.name,
        "predicted_class": cls,
        "confidence": conf,
        "health_status": h_status,
        "severity": sev,
        "recommendation": rec,
    }

    df_all = append_to_csv(record, CSV_PATH)

    st.markdown("---")
    st.subheader("📊 Current Prediction Log")
    st.dataframe(df_all, use_container_width=True)

else:
    df_all = load_csv(CSV_PATH)
    if not df_all.empty:
        st.subheader("📊 Current Prediction Log")
        st.dataframe(df_all, use_container_width=True)

# =====================================================
# 8. ANALYTICS / PLOTS FROM LOGGED PREDICTIONS
# =====================================================
st.markdown("---")
st.subheader("📈 Analytics from Logged Predictions")

df_log = load_csv(CSV_PATH)
if df_log.empty:
    st.info("No predictions logged yet. Upload an image to start building your dashboard.")
else:
    col_a, col_b = st.columns(2)

    # ---- Disease frequency ----
    with col_a:
        st.markdown("**Disease Frequency (Logged Images)**")
        disease_counts = df_log["predicted_class"].value_counts()
        st.bar_chart(disease_counts)

    # ---- Severity distribution ----
    with col_b:
        st.markdown("**Severity Distribution (Logged Images)**")
        sev_counts = df_log["severity"].value_counts()
        st.bar_chart(sev_counts)

    # ---- Filtered view by health status ----
    st.markdown("### 🔎 Filtered View")
    health_filter = st.selectbox(
        "Filter by Health Status",
        options=["All", "Healthy", "Diseased"],
        index=0,
    )

    if health_filter == "All":
        df_filtered = df_log
    else:
        df_filtered = df_log[df_log["health_status"] == health_filter]

    st.dataframe(df_filtered, use_container_width=True)

# =====================================================
# 9. STATIC MODEL EVALUATION (FROM YOUR REPORT)
# =====================================================
st.markdown("---")
st.subheader("📊  Model Evaluation (Validation Set)")

# Show global metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{STATIC_SUMMARY['accuracy']:.3f}")
# m2.metric("Macro Precision", f"{STATIC_SUMMARY['macro_precision']:.3f}")
# m3.metric("Macro Recall", f"{STATIC_SUMMARY['macro_recall']:.3f}")
# m4.metric("Macro F1-score", f"{STATIC_SUMMARY['macro_f1']:.3f}")

# st.caption(f"Evaluated on validation set with **{STATIC_SUMMARY['total_samples']}** samples.")

st.markdown("### Per-Class Metrics")

# Convert STATIC_REPORT to DataFrame for easier display
metrics_df = pd.DataFrame.from_dict(STATIC_REPORT, orient="index")
metrics_df.index.name = "class_name"
metrics_df = metrics_df.reset_index()

# Dropdown: all classes or single
options = ["All classes"] + list(metrics_df["class_name"])
selected = st.selectbox("Select class to inspect", options=options, index=0)

if selected == "All classes":
    st.dataframe(metrics_df, use_container_width=True)

    # Optional: bar chart of per-class F1
    st.markdown("**Per-Class F1-score**")
    chart_df = metrics_df.set_index("class_name")[["f1"]]
    st.bar_chart(chart_df)
else:
    row = metrics_df[metrics_df["class_name"] == selected].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{row['precision']:.3f}")
    c2.metric("Recall", f"{row['recall']:.3f}")
    c3.metric("F1-score", f"{row['f1']:.3f}")
    c4.metric("Support", int(row["support"]))

    st.write("Details:")
    st.table(row.to_frame().T)
