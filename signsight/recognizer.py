import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # show only TensorFlow errors

import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# Letters supported by the model (J and Z require motion and are excluded)
LETTER_PRED = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
]


@dataclass
class Prediction:
    """Holds the result of a single frame prediction."""

    letter: Optional[str]
    confidence: float
    frame: np.ndarray


class SignRecognizer:
    """Predicts ASL letters from images containing a single hand."""

    def __init__(self, model_path: str = "model.h5") -> None:
        self.model = load_model(model_path)
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def _extract_hand(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], np.ndarray]:
        """Detect the first hand in the frame and return the crop and annotated frame."""

        h, w, _ = frame.shape
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_landmarks = result.multi_hand_landmarks
        if not hand_landmarks:
            return False, None, frame

        handLMs = hand_landmarks[0]
        x_max, y_max, x_min, y_min = 0, 0, w, h
        for lm in handLMs.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_max = max(x_max, x)
            x_min = min(x_min, x)
            y_max = max(y_max, y)
            y_min = min(y_min, y)

        y_min = max(0, y_min - 20)
        y_max = min(h, y_max + 20)
        x_min = max(0, x_min - 20)
        x_max = min(w, x_max + 20)

        self.mp_drawing.draw_landmarks(frame, handLMs, self.mphands.HAND_CONNECTIONS)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        crop = frame[y_min:y_max, x_min:x_max]
        return True, crop, frame

    def predict(self, frame: np.ndarray) -> Prediction:
        """Return the predicted letter and confidence for a frame."""

        detected, crop, annotated = self._extract_hand(frame)
        if not detected or crop.size == 0:
            return Prediction(None, 0.0, annotated)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (28, 28))
        pixeldata = gray.reshape(1, 28, 28, 1) / 255.0
        prediction = self.model.predict(pixeldata, verbose=0)[0]
        idx = int(np.argmax(prediction))
        return Prediction(LETTER_PRED[idx], float(prediction[idx]), annotated)


def run_webcam(model_path: str = "model.h5") -> None:
    """Run real-time recognition using the default webcam."""

    recognizer = SignRecognizer(model_path)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = recognizer.predict(frame)
        if result.letter:
            text = f"Prediction: {result.letter} ({result.confidence * 100:.1f}%)"
            cv2.putText(result.frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(result.frame, "No hands detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Sign Language Recognition", result.frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
