import os
import tarfile
import tempfile
from typing import List
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

DATA_DIR = Path("data")
DATA_FILEPATH = Path("data") / "penguins.csv"


def get_root_dir() -> Path:
    """
    Returns the root directory of the project.
    """
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    else:
        return Path.cwd()


def get_data_from_file(data_filepath):
    """
    This function reads the data from the penguins.csv file.
    """
    root = get_root_dir()
    path = root / data_filepath
    return pd.read_csv(path)


def initialize_transformers(
    target_column: str,
    features_columns: List[str] = [
        "island",
    ],
) -> tuple[ColumnTransformer, ColumnTransformer]:
    """
    Initialize and return the target and feature transformers.

    Returns:
        tuple: A tuple containing the target transformer and
        the features transformer.
    """
    target_transformer = ColumnTransformer(
        transformers=[(target_column, OrdinalEncoder(), [0])]
    )

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"), StandardScaler()
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"), OneHotEncoder()
    )

    features_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                make_column_selector(dtype_exclude="object"),
            ),
            ("categorical", categorical_transformer, features_columns),
        ]
    )

    return target_transformer, features_transformer


def transform_target(
    target_transformer: ColumnTransformer,
    df_train: pd.DataFrame,
    df_validation: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transform the target variable.

    Args:
        target_transformer (ColumnTransformer): The transformer for the target
        variable.
        df_train (pd.DataFrame): The training dataframe.
        df_validation (pd.DataFrame): The validation dataframe.
        df_test (pd.DataFrame): The test dataframe.

    Returns:
        tuple: A tuple containing the transformed target variables for training, validation, and test sets.
    """
    y_train = target_transformer.fit_transform(
        np.array(df_train.species.values).reshape(-1, 1)
    )
    y_validation = target_transformer.transform(
        np.array(df_validation.species.values).reshape(-1, 1)
    )
    y_test = target_transformer.transform(
        np.array(df_test.species.values).reshape(-1, 1)
    )
    return y_train, y_validation, y_test


def drop_target(
    df_train: pd.DataFrame,
    df_validation: pd.DataFrame,
    df_test: pd.DataFrame,
    column_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Drop the target column from the dataframes.

    Args:
        df_train (pd.DataFrame): The training dataframe.
        df_validation (pd.DataFrame): The validation dataframe.
        df_test (pd.DataFrame): The test dataframe.
        column_name (str): The name of the target column to drop.

    Returns:
        tuple: A tuple containing the dataframes without the target column.
    """

    df_train = df_train.drop(column_name, axis=1)
    df_validation = df_validation.drop(column_name, axis=1)
    df_test = df_test.drop(column_name, axis=1)
    return df_train, df_validation, df_test


def transform_features(features_transformer, df_train, df_validation, df_test):
    """
    Transform the features.

    Args:
        features_transformer (ColumnTransformer): The transformer for the features.
        df_train (pd.DataFrame): The training dataframe.
        df_validation (pd.DataFrame): The validation dataframe.
        df_test (pd.DataFrame): The test dataframe.

    Returns:
        tuple: A tuple containing the transformed features for training, validation, and test sets.
    """
    X_train = features_transformer.fit_transform(df_train)
    X_validation = features_transformer.transform(df_validation)
    X_test = features_transformer.transform(df_test)
    return X_train, X_validation, X_test


def split_data(df):
    """
    Splits the data into three sets: train, validation and test.
    """

    df_train, temp = train_test_split(df, test_size=0.3)
    df_validation, df_test = train_test_split(temp, test_size=0.5)

    return df_train, df_validation, df_test


def save_baselines(base_directory, df_train, df_test):
    """
    During the data and quality monitoring steps, we will need baselines
    to compute constraints and statistics. This function saves the
    untransformed data to disk so we can use them as baselines later.
    """

    for split, data in [("train", df_train), ("test", df_test)]:
        baseline_path = Path(base_directory) / f"{split}-baseline"
        baseline_path.mkdir(parents=True, exist_ok=True)

        df = data.copy().dropna()

        # We want to save the header only for the train baseline
        # but not for the test baseline. We'll use the test baseline
        # to generate predictions later, and we can't have a header line
        # because the model won't be able to make a prediction for it.
        header = split == "train"
        df.to_csv(
            baseline_path / f"{split}-baseline.csv", header=header, index=False
        )


def save_splits(
    base_directory,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
):
    """
    This function concatenates the transformed features and the target
    variable, and saves each one of the split sets to disk.
    """

    train = np.concatenate((X_train, y_train), axis=1)
    validation = np.concatenate((X_validation, y_validation), axis=1)
    test = np.concatenate((X_test, y_test), axis=1)

    train_path = Path(base_directory) / "train"
    validation_path = Path(base_directory) / "validation"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(
        train_path / "train.csv", header=False, index=False
    )
    pd.DataFrame(validation).to_csv(
        validation_path / "validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(
        test_path / "test.csv", header=False, index=False
    )


def save_model(base_directory, target_transformer, features_transformer):
    """
    This function creates a model.tar.gz file that contains the two
    transformation pipelines we built to transform the data.
    """

    with tempfile.TemporaryDirectory() as directory:
        joblib.dump(
            target_transformer, os.path.join(directory, "target.joblib")
        )
        joblib.dump(
            features_transformer, os.path.join(directory, "features.joblib")
        )

        model_path = Path(base_directory) / "model"
        model_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(
            f"{str(model_path / 'model.tar.gz')}", "w:gz"
        ) as tar:
            tar.add(
                os.path.join(directory, "target.joblib"),
                arcname="target.joblib",
            )
            tar.add(
                os.path.join(directory, "features.joblib"),
                arcname="features.joblib",
            )


def preprocess_and_save_data(
    data_dir: str,
):
    """
    This function loads the supplied data, splits it, and transforms it.

    Args:
        df (pd.DataFrame): The input dataframe.
        base_directory (str): The base directory to save the processed data
        and models.
    """

    data_path = Path(data_dir) / "penguins.csv"
    df = get_data_from_file(data_filepath=data_path)

    target_transformer, features_transformer = initialize_transformers(
        target_column="species"
    )

    df_train_baseline, df_validation_baseline, df_test_baseline = split_data(
        df
    )

    y_train, y_validation, y_test = transform_target(
        target_transformer,
        df_train_baseline,
        df_validation_baseline,
        df_test_baseline,
    )

    df_train, df_validation, df_test = drop_target(
        df_train_baseline,
        df_validation_baseline,
        df_test_baseline,
        column_name="species",
    )

    X_train, X_validation, X_test = transform_features(
        features_transformer, df_train, df_validation, df_test
    )

    save_baselines(data_dir, df_train_baseline, df_test_baseline)
    save_splits(
        data_dir,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )
    save_model(data_dir, target_transformer, features_transformer)


def main():
    # TODO: to delete
    preprocess_and_save_data(data_dir=DATA_DIR)


if __name__ == "__main__":
    main()
