import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from preprocessing.filters import gaussian_filter

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Plant Leaf Disease Classification",
    layout="wide"
)

st.title("🌿 Plant Leaf Disease Classification")
st.write(
    "Aplikasi web untuk **klasifikasi penyakit daun tanaman tomat** "
    "menggunakan **CNN MobileNetV2**."
)

# =============================
# LOAD MODEL (CACHED)
# =============================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "model/model_mobilenetv2.h5"
    )
    return model

model = load_model()

# =============================
# CLASS LABELS (HARUS SESUAI TRAINING)
# =============================
CLASS_NAMES = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_healthy"
]

# =============================
# IMAGE PREPROCESS FUNCTION
# =============================
def preprocess_image(image):
    """
    Preprocess image for MobileNetV2
    """
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# =============================
# FILE UPLOADER
# =============================
uploaded_file = st.file_uploader(
    "Upload gambar daun tomat (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Convert RGB → BGR
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Optional pre-processing (Gaussian filter)
    processed_img = gaussian_filter(image_bgr)

    # Prepare for CNN
    input_tensor = preprocess_image(
        cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    )

    # Prediction
    predictions = model.predict(input_tensor)
    confidence = np.max(predictions)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    # =============================
    # DISPLAY
    # =============================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image_np, use_container_width=True)

    with col2:
        st.subheader("Prediction Result")
        st.markdown(f"### 🧠 **{predicted_class}**")
        st.markdown(
            f"**Confidence:** `{confidence * 100:.2f}%`"
        )

        if predicted_class == "Tomato_healthy":
            st.success("Daun dalam kondisi sehat ✅")
        else:
            st.warning("Daun terdeteksi mengalami penyakit ⚠️")

else:
    st.info("Silakan upload gambar daun untuk memulai klasifikasi.")

