import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd

# Try to load the model, with error handling
try:
    model = load_model('model.keras')
except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure your model file exists and is correctly named.")
        exit(1)

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape

letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# For controlling prediction frequency
last_prediction_time = time.time()
prediction_interval = 1.0  # seconds between predictions
current_prediction = ""
confidence = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    # Process frame for hand detection
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks
    
    # Initialize bounding box coordinates
    x_min, y_min, x_max, y_max = 0, 0, w, h
    hand_detected = False
    
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min = max(0, y_min - 20)
            y_max = min(h, y_max + 20)
            x_min = max(0, x_min - 20)
            x_max = min(w, x_max + 20)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            hand_detected = True
            
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
    
    # Perform prediction at regular intervals if hand is detected
    current_time = time.time()
    if hand_detected and (current_time - last_prediction_time) > prediction_interval:
        try:
            # Create a copy of the frame for analysis
            analysisframe = frame.copy()
            
            # Convert to grayscale and crop to hand region
            analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
            analysisframe = analysisframe[y_min:y_max, x_min:x_max]
            
            # Resize to 28x28 for the model
            analysisframe = cv2.resize(analysisframe, (28, 28))
            
            # Convert to format needed for model
            nlist = []
            rows, cols = analysisframe.shape
            for i in range(rows):
                for j in range(cols):
                    k = analysisframe[i, j]
                    nlist.append(k)
            
            datan = pd.DataFrame(nlist).T
            colname = []
            for val in range(784):
                colname.append(val)
            datan.columns = colname
            
            pixeldata = datan.values
            pixeldata = pixeldata / 255
            pixeldata = pixeldata.reshape(-1, 28, 28, 1)
            
            # Make prediction
            prediction = model.predict(pixeldata, verbose=0)
            predarray = np.array(prediction[0])
            letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
            predarrayordered = sorted(predarray, reverse=True)
            high1 = predarrayordered[0]
            
            # Update current prediction
            for key, value in letter_prediction_dict.items():
                if value == high1:
                    current_prediction = key
                    confidence = value
                    break
            
            last_prediction_time = current_time
            
        except Exception as e:
            print(f"Error during prediction: {e}")
    
    # Display status on the frame
    if hand_detected:
        if current_prediction:
            prediction_text = f"Prediction: {current_prediction} ({confidence*100:.1f}%)"
            cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Display "No hands detected" when no hands are visible
        cv2.putText(frame, "No hands detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # Reset current prediction when no hands are detected
        if current_time - last_prediction_time > 2.0:  # After 2 seconds of no hands
            current_prediction = ""
            confidence = 0
    
    # Display the frame
    cv2.imshow("Sign Language Recognition", frame)
    
    # Check for ESC key
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()