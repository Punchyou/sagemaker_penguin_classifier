from pathlib import Path
from datetime import datetime
from sagemaker import Session
from sagemaker.workflow.parameters import ParameterString
import os

LOCAL_MODE = True
IS_APPLE_M_CHIP = True


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_FILENAME = Path("data") / "model" / f"model_{TIMESTAMP}.keras"
DATA_DIR = Path("data")
TRAIN_CSV_PATH = Path("data") / "train" / "train.csv"
VALIDATION_CSV_PATH = Path("data") / "validation" / "validation.csv"
PREPROCESSING_FILENAME = Path("preprocess_data") / "preprocessor.py"

EPOCHS = 50
BATCH_SIZE = 32

SAGEMAKER_SESSION_CONFIG = {
    "framework_version": "1.2-1",
    "instance_type": "ml.m5.xlarge",
    "instance_count": 1,
    "session": Session(),
}

S3_LOCATION = os.getenv("S3_LOCATION")
DATASET_LOCATION = ParameterString(
    name="dataset_location",
    default_value=f"s3://{S3_LOCATION}/data",
)
AWS_ROLE = os.getenv("AWS_ROLE")
