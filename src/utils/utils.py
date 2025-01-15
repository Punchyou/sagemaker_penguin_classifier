from pathlib import Path
from typing import Union
import pandas as pd

from utils.S3Uploader import S3Uploader
from .logging_utils import Logger
from constants import S3_BUCKET, S3_PREPROCESSOR_UTILS_KEY, S3_LOGGING_UTILS_KEY

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


# upload to s3 utils
def upload_file_to_s3(local_path: Union[str, Path], bucket: str, s3_key: str) -> None:
    """
    Upload a file to an S3 bucket without checking if it exists.

    Parameters
    ----------
    local_path : str
        The local path of the file to upload.
    bucket : str
        The name of the S3 bucket.
    s3_key : str
        The key of the file in the S3 bucket.
    """
    if not bucket:
        raise ValueError("bucket must be specified.")
    uploader = S3Uploader()
    uploader.upload_file(local_path, bucket, s3_key)
    logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")


def upload_preprocessing_dependencies_to_s3() -> None:
    """
    Upload the preprocessing dependencies to the specified S3 bucket.

    Parameters
    ----------
    bucket : str
        The name of the S3 bucket.
    s3_destination : str
        The destination in the S3 bucket.
    """
    upload_file_to_s3(
        local_path=Path(__file__).resolve().parent / "preprocessor_utils.py",
        bucket=S3_BUCKET,  # type: ignore
        s3_key=S3_PREPROCESSOR_UTILS_KEY,
    )
    upload_file_to_s3(
        local_path=Path(__file__).resolve().parent / "logging_utils.py",
        bucket=S3_BUCKET,  # type: ignore
        s3_key=S3_LOGGING_UTILS_KEY,
    )
