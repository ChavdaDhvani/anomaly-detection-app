import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

st.title("Anomaly Detection App")

# Load model once at start
model = tf.keras.models.load_model('keras_model.h5', compile=False)
with open('labels.txt', 'r') as f:
    labels = f.read().splitlines()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize to [0,1]
    return np.expand_dims(image, axis=0)

# File uploader section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_label = labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    st.write(f"Prediction: {predicted_label} ({confidence*100:.2f}%)")

# Webcam section
if st.button("Use Webcam"):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            processed_image = preprocess_image(img_pil)
            prediction = model.predict(processed_image)
            predicted_label = labels[np.argmax(prediction)]
            confidence = np.max(prediction)
            
            # Overlay prediction text on frame
            cv2.putText(frame, f"{predicted_label} ({confidence*100:.2f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            
            stframe.image(frame, channels="BGR")

            # Break on streamlit stop (this is a workaround since OpenCV waitKey won't work well here)
            if st.button("Stop Webcam"):
                break
        cap.release()
