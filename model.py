# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
import pandas as pd


# Load the Sign Language MNIST dataset
# Make sure the CSV files are in the same directory as the script
train_df = pd.read_csv("sign_mnist_train.csv")  # Training data
test_df = pd.read_csv("sign_mnist_test.csv")  # Testing data
print(train_df.head(10))  # Display first 10 rows of training data

# Separate labels from features
y_train = train_df.pop('label')  # Extract training labels
y_test = test_df.pop('label')  # Extract testing labels

# Convert labels to one-hot encoded format (required for categorical classification)
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

# Extract pixel values as features
x_train = train_df.values  # Training features
x_test = test_df.values  # Testing features

# Normalize pixel values to 0-1 range (improves training stability)
x_train = x_train / 255
x_test = x_test / 255

# Reshape data to 28x28 pixel images with 1 channel (grayscale)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Set up data augmentation to artificially increase training data variety
# This helps prevent overfitting and improves model generalization
datagen = ImageDataGenerator(
        featurewise_center=False,  # Don't adjust mean of all images to 0
        samplewise_center=False,  # Don't adjust each sample mean to 0
        featurewise_std_normalization=False,  # Don't divide inputs by std of dataset
        samplewise_std_normalization=False,  # Don't divide each input by its std
        zca_whitening=False,  # Don't apply ZCA whitening
        rotation_range=10,  # Randomly rotate images by up to 10 degrees
        zoom_range = 0.1,  # Randomly zoom in/out by up to 10%
        width_shift_range=0.1,  # Randomly shift horizontally by up to 10%
        height_shift_range=0.1,  # Randomly shift vertically by up to 10%
        horizontal_flip=True,  # Randomly flip images horizontally
        vertical_flip=False)  # Don't flip vertically (would create invalid signs)

# Fit the data generator to the training data
datagen.fit(x_train)

# Initialize the sequential model
model = Sequential()

# Set up callbacks for training optimization
# Reduce learning rate when validation accuracy plateaus
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=2,  # Wait 2 epochs before reducing learning rate
    verbose=1,  # Print messages
    factor=0.5,  # Reduce learning rate by half
    min_lr=0.00001  # Minimum learning rate
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,  # Stop if no improvement for 5 epochs
    restore_best_weights=True  # Keep the best model weights
)

# Build the CNN model architecture
model = Sequential()

# First convolutional block
model.add(Conv2D(75, (3,3), strides=1, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(BatchNormalization())  # Normalize activations for better training stability
model.add(MaxPool2D((2,2), strides=2, padding='same'))  # Reduce spatial dimensions

# Second convolutional block
model.add(Conv2D(50, (3,3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))  # Randomly drop 20% of connections to prevent overfitting
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding='same'))

# Third convolutional block
model.add(Conv2D(25, (3,3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding='same'))

# Flatten the 3D feature maps to 1D feature vectors
model.add(Flatten())

# Fully connected layers for classification
model.add(Dense(units=512, activation='relu'))  # Hidden layer with 512 neurons
model.add(Dropout(0.3))  # Randomly drop 30% of connections to prevent overfitting
model.add(Dense(units=24, activation='softmax'))  # Output layer with 24 classes (ASL letters)

# Compile the model with appropriate loss function and optimizer
model.compile(
    optimizer='adam',  # Adam optimizer for adaptive learning rate
    loss='categorical_crossentropy',  # Standard loss for multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)

# Display model architecture summary
model.summary()

# Train the model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),  # Use data generator for augmentation
    epochs=20,  # Train for 20 epochs
    validation_data=(x_test, y_test),  # Validate on test data
    callbacks=[learning_rate_reduction, early_stopping]  # Use callbacks for optimization
)

# Save the trained model to a file
model.save('model.h5')
