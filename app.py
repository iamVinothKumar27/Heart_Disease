import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('/Users/t.s.vinoth/Desktop/Heart_Disease_Pred/model/heard_disease_predictor.h5')

st.title("🫀 Zenith Disease Predictor")
st.subheader("Upload an ECG image (JPG, PNG)")

uploaded_file = st.file_uploader("Choose an ECG image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded ECG", use_container_width=True)

    # ✅ Resize to expected shape (224, 224)
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.reshape(image_array, (1, 224, 224, 3))  # Shape: (1, 224, 224, 3)

    if st.button("Predict"):
        prediction = model.predict(image_array)
        result = "High Risk ⚠️" if prediction[0][0] > 0.5 else "Low Risk ✅"
        st.success(f"Prediction: {result}")
