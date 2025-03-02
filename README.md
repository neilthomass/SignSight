# Sign Language Recognition System

A real-time American Sign Language (ASL) recognition system using computer vision and deep learning.

## Overview

This project uses a convolutional neural network (CNN) to recognize American Sign Language (ASL) hand gestures in real-time through a webcam. The system can identify 24 static ASL signs (A-Y, excluding J and Z which require motion).

![Sign Language Recognition Demo](https://i.imgur.com/example.gif)

## Features

- **Real-time Recognition**: Continuously detects and classifies hand signs
- **Visual Feedback**: Displays prediction results directly on the video feed
- **Hand Tracking**: Uses MediaPipe to detect and track hand landmarks
- **User-friendly Interface**: Clear visual indicators for hand detection status

## Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- MediaPipe
- NumPy
- Pandas

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sign-language-recognition.git
   cd sign-language-recognition
   ```

2. Install the required packages:
   ```
   pip install tensorflow opencv-python mediapipe numpy pandas
   ```

3. Download the pre-trained model or train your own using the provided code.

## Usage

1. Run the main application:
   ```
   python main.py
   ```

2. Position your hand in the camera's field of view.

3. Make ASL hand signs, and the system will display the recognized letter.

4. Press ESC to exit the application.

## How It Works

1. **Hand Detection**: MediaPipe's hand tracking solution detects hands in the video feed.

2. **Image Processing**: The hand region is cropped, converted to grayscale, and resized to 28x28 pixels.

3. **Classification**: The processed image is fed into a CNN model trained on the Sign Language MNIST dataset.

4. **Display**: The predicted letter and confidence score are displayed on the screen.

## Model Architecture

The CNN model consists of:
- 3 convolutional layers with batch normalization and max pooling
- Dropout layers for regularization
- Dense layers for classification
- Trained on the Sign Language MNIST dataset

## Training Your Own Model

If you want to train your own model:

1. Download the Sign Language MNIST dataset from Kaggle.

2. Use the `model.py` script to train the model:
   ```
   python model.py
   ```

3. The trained model will be saved as `smnist.h5`.

## Limitations

- The system recognizes static hand gestures only (not including signs that require motion like J and Z).
- Performance may vary depending on lighting conditions and background.
- The model works best with a clear, uncluttered background.

## Future Improvements

- Add support for dynamic gestures (including J and Z)
- Implement word and sentence recognition
- Improve robustness to different lighting conditions
- Add a user interface for model training and configuration

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data) - Thanks to the dataset creators and Kaggle for hosting this valuable resource
- MediaPipe for hand tracking
- TensorFlow and Keras for deep learning framework # ComputerVisonASL
