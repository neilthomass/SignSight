import os
# Set TensorFlow logging level to only show errors, suppressing warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

# Import necessary libraries
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd

# Try to load the pre-trained model, with error handling
try:
    model = load_model('model.h5')  # Load the ASL sign recognition model
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure your model file exists and is correctly named.")
    exit(1)  # Exit if model loading fails

# Initialize MediaPipe hand detection components
mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)  # Initialize webcam (0 is usually the default camera)

# Read a frame to get dimensions
_, frame = cap.read()
h, w, c = frame.shape  # Height, width, and channels of the frame

# Define the ASL letters that our model can predict
# Note: J and Z are excluded as they require motion
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Variables for controlling prediction frequency and storing results
last_prediction_time = time.time()  # Track when the last prediction was made
prediction_interval = 1.0  # Seconds between predictions (to avoid excessive processing)
current_prediction = ""  # Store the current predicted letter
confidence = 0  # Store the confidence level of the prediction

# Main processing loop
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    # Convert frame to RGB for MediaPipe processing (MediaPipe uses RGB, OpenCV uses BGR)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)  # Process the frame to detect hands
    hand_landmarks = result.multi_hand_landmarks  # Extract hand landmarks if detected
    
    # Initialize bounding box coordinates for the hand
    x_min, y_min, x_max, y_max = 0, 0, w, h
    hand_detected = False
    
    # If hands are detected, process the landmarks
    if hand_landmarks:
        for handLMs in hand_landmarks:
            # Reset bounding box for each hand
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            
            # Find the bounding box that contains all hand landmarks
            for lm in handLMs.landmark:
                # Convert normalized coordinates to pixel coordinates
                x, y = int(lm.x * w), int(lm.y * h)
                # Update bounding box boundaries
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            
            # Add padding to the bounding box (20 pixels on each side)
            y_min = max(0, y_min - 20)  # Ensure we don't go outside the frame
            y_max = min(h, y_max + 20)
            x_min = max(0, x_min - 20)
            x_max = min(w, x_max + 20)
            
            # Draw the bounding box around the hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            hand_detected = True
            
            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)
    
    # Perform prediction at regular intervals if a hand is detected
    current_time = time.time()
    if hand_detected and (current_time - last_prediction_time) > prediction_interval:
        try:
            # Create a copy of the frame for analysis
            analysisframe = frame.copy()
            
            # Convert to grayscale and crop to hand region
            analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
            analysisframe = analysisframe[y_min:y_max, x_min:x_max]
            
            # Resize to 28x28 pixels to match the model's expected input size
            analysisframe = cv2.resize(analysisframe, (28, 28))
            
            # Convert the 2D image to a 1D list of pixel values
            nlist = []
            rows, cols = analysisframe.shape
            for i in range(rows):
                for j in range(cols):
                    k = analysisframe[i, j]
                    nlist.append(k)
            
            # Convert to DataFrame format (similar to how the model was trained)
            datan = pd.DataFrame(nlist).T
            colname = []
            for val in range(784):  # 28x28 = 784 pixels
                colname.append(val)
            datan.columns = colname
            
            # Normalize pixel values to 0-1 range and reshape for the model
            pixeldata = datan.values
            pixeldata = pixeldata / 255  # Normalize to 0-1 range
            pixeldata = pixeldata.reshape(-1, 28, 28, 1)  # Reshape to match model input shape
            
            # Make prediction using the model
            prediction = model.predict(pixeldata, verbose=0)
            predarray = np.array(prediction[0])
            
            # Create a dictionary mapping letters to their prediction probabilities
            letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
            
            # Find the letter with the highest probability
            predarrayordered = sorted(predarray, reverse=True)
            high1 = predarrayordered[0]  # Highest probability
            
            # Update current prediction and confidence
            for key, value in letter_prediction_dict.items():
                if value == high1:
                    current_prediction = key
                    confidence = value
                    break
            
            # Update the last prediction time
            last_prediction_time = current_time
            
        except Exception as e:
            print(f"Error during prediction: {e}")
    
    # Display prediction status on the frame
    if hand_detected:
        if current_prediction:
            # Show the predicted letter and confidence percentage
            prediction_text = f"Prediction: {current_prediction} ({confidence*100:.1f}%)"
            cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Display "No hands detected" when no hands are visible
        cv2.putText(frame, "No hands detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Reset current prediction when no hands are detected for a while
        if current_time - last_prediction_time > 2.0:  # After 2 seconds of no hands
            current_prediction = ""
            confidence = 0
    
    # Display the processed frame
    cv2.imshow("Sign Language Recognition", frame)
    
    # Check for ESC key press to exit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ASCII code for ESC key
        print("Escape hit, closing...")
        break

# Release resources
cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
