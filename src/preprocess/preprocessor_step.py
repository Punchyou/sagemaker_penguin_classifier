import boto3
import pandas as pd
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
import os

from constants import (
    AWS_ROLE,
    BASE_JOB_NAME,
    DATA_FILE,
    INPUT_PREFIX,
    OUTPUT_BUCKET_NAME,
    OUTPUT_PREFIX,
    S3_BUCKET,
    SAGEMAKER_SESSION_CONFIG,
)


def upload_to_s3(file_path):
    session = sagemaker.Session()
    s3_client = session.boto_session.client("s3")

    # Get just the filename from the path
    file_name = os.path.basename(file_path)

    # Construct the S3 key
    s3_key = f"{INPUT_PREFIX}/{file_name}"
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"

    try:
        # Check if file exists
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        print(f"File already exists in S3: {s3_uri}")
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            # File doesn't exist, upload it
            print(f"Uploading file to S3: {s3_uri}")
            session.upload_data(
                path=file_name, bucket=S3_BUCKET, key_prefix=INPUT_PREFIX
            )
        else:
            print(f"Error checking file existence in S3: {e}")
            raise e
    return s3_uri


def run_processing_job():
    # Create and upload dummy data
    # data_file = create_dummy_data()

    s3_data_path = upload_to_s3(DATA_FILE)

    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=AWS_ROLE,
        instance_type=SAGEMAKER_SESSION_CONFIG["instance_type"],
        instance_count=int(SAGEMAKER_SESSION_CONFIG["instance_count"]),
        base_job_name=BASE_JOB_NAME,  # data-processing),
    )

    sklearn_processor.run(
        code="src/preprocess/preprocessor.py",
        inputs=[
            ProcessingInput(
                source=s3_data_path,
                destination="/opt/ml/processing/input/data",
                input_name="data",
            )
        ],
        # outputs=[
        #     ProcessingOutput(
        #         output_name="processed_data",
        #         source="/opt/ml/processing/output",
        #         destination=f"s3://{OUTPUT_BUCKET_NAME}/{OUTPUT_PREFIX}",
        #     )
        # ],
        logs=True,
    )
