import os
from pathlib import Path
from typing import Union
import pandas as pd
import sagemaker

from utils.S3Uploader import S3Uploader
from .logging_utils import Logger
from constants import (
    CSV_FILENAME,
    S3_BUCKET,
    S3_DATA_PREFIX,
    S3_PREPROCESSOR_UTILS_KEY,
    S3_LOGGING_UTILS_KEY,
)

logger = Logger().get_logger()


def read_data_from_input_csv_files(base_directory: Union[str, Path]) -> pd.DataFrame:
    """
    This function reads every CSV file available and concatenates
    them into a single dataframe. Returns a sample of the dataframe.

    Parameters
    ----------
    base_directory : str or Path
        The base directory where the CSV files are stored.

    Returns
    -------
    pd.DataFrame
        A sample of the concatenated dataframe.
    """

    input_directory = Path(base_directory) / "input"
    files = [file for file in input_directory.glob("*.csv")]

    if len(files) == 0:
        raise ValueError(f"The are no CSV files in {str(input_directory)}/")

    raw_data = [pd.read_csv(file) for file in files]
    df = pd.concat(raw_data)

    # Shuffle the data
    return df.sample(frac=1, random_state=42)
