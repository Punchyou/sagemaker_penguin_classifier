import numpy as np
import pandas as pd
import fire

from pathlib import Path
from sklearn.metrics import accuracy_score

from utils.logging import Logger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


logger = Logger().get_logger()


def train(
    epochs: int,
    batch_size: int,
    train_csv_path: Path,
    validation_csv_path: Path,
) -> Sequential:
    """
    Train a simple neural network model using the provided data.

    The neural network consists of an input layer with 10 units, a hidden
    layer with 8 units, and an output layer with 3 units.

    Parameters
    ----------
    epochs : int
        The number of epochs to train the model.
    batch_size : int
        The batch size for training.
    train_csv_path : str
        The path to the training CSV file.
    validation_csv_path : str
        The path to the validation CSV file.
    """
    X_train = pd.read_csv(train_csv_path)
    y_train = X_train[X_train.columns[-1]]
    X_train.drop(X_train.columns[-1], axis=1, inplace=True)

    X_validation = pd.read_csv(validation_csv_path)
    y_validation = X_validation[X_validation.columns[-1]]
    X_validation.drop(X_validation.columns[-1], axis=1, inplace=True)

    model = Sequential(
        [
            Dense(10, input_shape=(X_train.shape[1],), activation="relu"),
            Dense(8, activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    predictions = np.argmax(model.predict(X_validation), axis=-1)
    logger.info(
        f"Validation accuracy: {accuracy_score(y_validation, predictions)}"
    )
    return model


def save_model(model: Sequential, model_filename: Path) -> None:
    """
    Save the trained model to the specified directory.

    Parameters
    ----------
    model : keras.models.Sequential
        The trained model.
    model_dir_path : str
        The directory path where the model artifacts will be saved.
    """

    model.save(model_filename)
    logger.info(f"Model artifacts saved as: {model_filename}")


def main(
    epochs: int,
    batch_size: int,
    train_csv_path: Path,
    validation_csv_path: Path,
    model_filename: Path,
) -> None:
    """
    Main function to train the model and save it.

    Parameters
    ----------
    epochs : int, optional
        The number of epochs to train the model, by default 50.
    batch_size : int, optional
        The batch size for training, by default 32.
    """
    trained_model = train(
        epochs=epochs,
        batch_size=batch_size,
        train_csv_path=train_csv_path,
        validation_csv_path=validation_csv_path,
    )
    save_model(trained_model, model_filename=model_filename)


if __name__ == "__main__":
    fire.Fire(main)
