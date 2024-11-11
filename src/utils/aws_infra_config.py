import logging
import boto3
from sagemaker.workflow.pipeline_context import (
    LocalPipelineSession,
    PipelineSession,
)
import sagemaker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_sagemaker_session():
    """
    Initialize and return a SageMaker session.

    Returns
    -------
    sagemaker.session.Session
        A SageMaker session object.
    """
    return sagemaker.session.Session()


def get_sagemaker_client():
    """
    Initialize and return a SageMaker client.

    Returns
    -------
    boto3.client
        A SageMaker client object.
    """
    return boto3.client("sagemaker")


def get_iam_client():
    """
    Initialize and return an IAM client.

    Returns
    -------
    boto3.client:
        An IAM client object.
    """
    return boto3.client("iam")


def get_region():
    """
    Get the AWS region name.

    Returns
    -------
    str:
        The AWS region name.
    """
    return boto3.Session().region_name


def get_sm_config_with_local_mode(local_mode, apple_m_chip) -> dict:
    """
    Get the SageMaker configuration settings.

    Parameters
    ----------
    local_mode : bool
        Whether to run the pipeline in local mode.
    apple_m_chip : bool
        Whether current machine is built with Apple M1 chip.
    """
    return {
        "session": LocalPipelineSession() if local_mode else PipelineSession(),
        "instance_type": "local" if local_mode else "ml.m5.xlarge",
        "image": (
            "sagemaker-tensorflow-toolkit-local"
            if local_mode and apple_m_chip
            else None
        ),
        "framework_version": "2.11",
        "py_version": "py310",
    }
