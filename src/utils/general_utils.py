from pathlib import Path
import pandas as pd


def get_root_dir() -> Path:
    """
    Returns the root directory of the project.

    Returns
    -------
    Path
        The root directory of the project.
    """
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    else:
        return Path.cwd()


def get_data_from_file(data_filepath) -> pd.DataFrame:
    """
    Reads the data from the specified CSV file.

    Parameters
    ----------
    data_filepath : str
        The relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The data read from the CSV file.
    """
    root = get_root_dir()
    path = root / data_filepath
    return pd.read_csv(path)


def read_data_from_input_csv_files(base_directory):
    """
    This function reads every CSV file available and concatenates
    them into a single dataframe.
    """

    input_directory = Path(base_directory) / "input"
    files = [file for file in input_directory.glob("*.csv")]

    if len(files) == 0:
        raise ValueError(f"The are no CSV files in {str(input_directory)}/")

    raw_data = [pd.read_csv(file) for file in files]
    df = pd.concat(raw_data)

    # Shuffle the data
    return df.sample(frac=1, random_state=42)
