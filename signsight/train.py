import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd


def train_model(train_csv: str = "sign_mnist_train.csv", test_csv: str = "sign_mnist_test.csv") -> None:
    """Train the CNN on the Sign Language MNIST dataset and save the model."""

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    y_train = train_df.pop("label")
    y_test = test_df.pop("label")

    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.transform(y_test)

    x_train = train_df.values / 255.0
    x_test = test_df.values / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    learning_rate_reduction = ReduceLROnPlateau(
        monitor="val_accuracy", patience=2, verbose=1, factor=0.5, min_lr=1e-5
    )
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model = Sequential([
        Conv2D(75, (3, 3), padding="same", activation="relu", input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding="same"),
        Conv2D(50, (3, 3), padding="same", activation="relu"),
        Dropout(0.2),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding="same"),
        Conv2D(25, (3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding="same"),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.3),
        Dense(24, activation="softmax"),
    ])

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=[learning_rate_reduction, early_stopping],
    )

    model.save("model.h5")


if __name__ == "__main__":
    train_model()
