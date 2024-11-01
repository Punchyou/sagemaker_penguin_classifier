import boto3
import sagemaker
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from aws_infra_config import get_sagemaker_config
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sagemaker_session():
    """
    Initialize and return a SageMaker session.

    Returns:
        sagemaker.session.Session: A SageMaker session object.
    """
    return sagemaker.session.Session()


def get_sagemaker_client():
    """
    Initialize and return a SageMaker client.

    Returns:
        boto3.client: A SageMaker client object.
    """
    return boto3.client("sagemaker")


def get_iam_client():
    """
    Initialize and return an IAM client.

    Returns:
        boto3.client: An IAM client object.
    """
    return boto3.client("iam")


def get_region():
    """
    Get the AWS region name.

    Returns:
        str: The AWS region name.
    """
    return boto3.Session().region_name


def main():
    logger.info("Starting the SageMaker application")
    sagemaker_config = get_sagemaker_config()
    sagemaker_session = get_sagemaker_session()
    sagemaker_client = get_sagemaker_client()
    iam_client = get_iam_client()
    region = get_region()
    logger.info("Region: %s", region)


if __name__ == "__main__":
    main()
