from pathlib import Path
from datetime import datetime
from sagemaker import Session
from sagemaker.workflow.parameters import ParameterString
from sagemaker.local import LocalSession

import os

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Local config
LOCAL_MODE = False
IS_APPLE_M_CHIP = True

# local files
DATA_DIR = Path("data")
DATA_FILE = Path(__file__).resolve().parent.parent / "src" / "data" / "penguins.csv"

# sagemaker container paths
TRAIN_CSV_PATH = "/opt/ml/data/train/train.csv"
VALIDATION_CSV_PATH = "/opt/ml/data/validation/validation.csv"
TEST_CSV_PATH = "/opt/ml/data/test/test.csv"
TRAIN_BASELINE_CSV_PATH = "/opt/ml/data/train-baseline/train-baseline.csv"
TEST_BASELINE_CSV_PATH = "/opt/ml/data/test-baseline/test-baseline.csv"
# Files to be used as inputs/dependencies
# intended to be downloaded from S3 to the SageMaker container
INPUT_DATA = "/opt/ml/processing/input"
UTILS_DIR = "/opt/ml/processing/input/code/"


# model
EPOCHS = 50
BATCH_SIZE = 32
MODEL_FILENAME = f"/opt/ml/data/model/model_{TIMESTAMP}.keras"

# script filepaths
PREPROCESSING_FILENAME = "preprocess/preprocessor.py"


# AWS
# General
AWS_ROLE = os.getenv("AWS_ROLE")

local_session = LocalSession()
local_session.config = {"local": {"local_code": True}}

SAGEMAKER_SESSION_CONFIG = {
    "framework_version": "1.2-1",
    "instance_type": "local" if LOCAL_MODE else "ml.m5.large",
    "instance_count": 1,
    "session": local_session if LOCAL_MODE else Session(),
    "image": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
}

# dependencies
S3_LOCATION = os.getenv("S3_LOCATION")
S3_BUCKET = os.getenv("S3_BUCKET")
CSV_FILENAME = "penguins.csv"
S3_DATA_PREFIX = "data"
S3_DATASET_LOCATION = ParameterString(
    name="dataset_location",
    default_value=f"{S3_LOCATION}/data/penguins.csv",
)


S3_UTILS_LOCATION = ParameterString(
    name="preprocessor_utils_location",
    default_value=f"{S3_LOCATION}/input",
)

# preprocessing dependencies
S3_PREPROCESSOR_UTILS_KEY = "utils/preprocessor_utils.py"
S3_LOGGING_UTILS_KEY = "utils/logging_utils.py"

S3_TRAIN_DATA_DESTINATION = f"{S3_LOCATION}/preprocessing/train"
S3_VALIDATION_DATA_DESTINATION = f"{S3_LOCATION}/preprocessing/validation"
S3_TEST_DATA_DESTINATION = f"{S3_LOCATION}/preprocessing/test"
S3_TRAIN_BASELINE_DATA_DESTINATION = f"{S3_LOCATION}/preprocessing/train-baseline"
S3_TEST_BASELINE_DATA_DESTINATION = f"{S3_LOCATION}/preprocessing/test-baseline"
S3_MODEL_DESTINATION = f"{S3_LOCATION}/preprocessing/model"

IS_APPLE_M_CHIP = True
AWS_DEFAULT_REGION = "us-east-1"
# TODO: Update the S3 location
DATA_FILE = Path("data/penguins.csv")
INPUT_PREFIX = "data"
BASE_JOB_NAME = "data-processing"
OUTPUT_BUCKET_NAME = "?"
OUTPUT_PREFIX = "output"
