import boto3
import os
from .logging_utils import Logger

logger = Logger().get_logger()

class S3Uploader:
    def __init__(self):
        self.s3_client = boto3.client("s3")

    def file_exists(self, bucket, key):
        """
        Check if a file exists in an S3 bucket.

        Parameters
        ----------
        bucket : str
            The name of the S3 bucket.
        key : str
            The key of the file in the S3 bucket.

        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except self.s3_client.exceptions.ClientError:
            return False

    def upload_file_if_not_exists(self, local_path, bucket, s3_key):
        """
        Upload a file to an S3 bucket if it does not already exist.

        Parameters
        ----------
        local_path : str
            The local path of the file to upload.
        bucket : str
            The name of the S3 bucket.
        s3_key : str
            The key of the file in the S3 bucket.

        Returns
        -------
        None
        """
        if not self.file_exists(bucket, s3_key):
            self.s3_client.upload_file(local_path, bucket, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")
        else:
            logger.info(f"File s3://{bucket}/{s3_key} already exists, skipping upload.")

    def upload_file(self, local_path, bucket, s3_key):
        """
        Upload a file to an S3 bucket without checking if it already exists.

        Parameters
        ----------
        local_path : str
            The local path of the file to upload.
        bucket : str
            The name of the S3 bucket.
        s3_key : str
            The key of the file in the S3 bucket.

        Returns
        -------
        None
        """
        self.s3_client.upload_file(local_path, bucket, s3_key)
        logger.info(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")

    def upload_directory_if_not_exists(self, local_path, bucket, s3_prefix):
        """
        Upload a directory to an S3 bucket if the files do not already exist.

        Parameters
        ----------
        local_path : str
            The local path of the directory to upload.
        bucket : str
            The name of the S3 bucket.
        s3_prefix : str
            The prefix for the files in the S3 bucket.

        Returns
        -------
        None
        """
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_path)
                s3_key = os.path.join(s3_prefix, relative_path)
                if not self.file_exists(bucket, s3_key):
                    self.s3_client.upload_file(local_file_path, bucket, s3_key)
                    logger.info(f"Uploaded {local_file_path} to s3://{bucket}/{s3_key}")
                else:
                    logger.info(
                        f"File s3://{bucket}/{s3_key} already exists, skipping upload."
                    )
