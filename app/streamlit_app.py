import cv2
import numpy as np
import streamlit as st

from signsight import SignRecognizer

st.title("SignSight")

recognizer = SignRecognizer("model.h5")

st.write("Use the camera below to capture an ASL hand sign.")
image = st.camera_input("Take a picture")

if image is not None:
    file_bytes = np.frombuffer(image.getvalue(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    result = recognizer.predict(frame)
    st.image(cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB))
    if result.letter:
        st.success(f"Prediction: {result.letter} ({result.confidence * 100:.1f}%)")
    else:
        st.error("No hand detected")
