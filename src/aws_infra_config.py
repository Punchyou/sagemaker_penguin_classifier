import os
import logging
import sagemaker
from sagemaker.workflow.pipeline_context import (
    PipelineSession,
    LocalPipelineSession,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def is_local_mode() -> bool:
    """
    Determines if the script is running in local mode.
    """
    return os.getenv("LOCAL_MODE", "True") == "True"


def is_apple_m_chip() -> bool:
    """
    Determines if the script is running on an Apple M1 chip.
    """
    return os.getenv("IS_APPLE_M_CHIP", "False") == "True"


def get_pipeline_session(local_mode: bool) -> PipelineSession:
    """
    Returns the appropriate pipeline session based on the environment.
    """
    return PipelineSession() if not local_mode else None


def get_config(
    local_mode: bool, apple_m_chip: bool, pipeline_session: PipelineSession
) -> dict:
    """
    Returns the SageMaker configuration based on the environment.
    """
    return {
        "session": LocalPipelineSession() if local_mode else pipeline_session,
        "instance_type": "local" if local_mode else "ml.m5.xlarge",
        "image": (
            "sagemaker-tensorflow-toolkit-local"
            if local_mode and apple_m_chip
            else None
        ),
        "framework_version": "2.11",
        "py_version": "py310",
    }


def get_sagemaker_config() -> dict:
    """
    Returns the SageMaker configuration based on the environment.
    """
    local_mode = is_local_mode()
    logging.info("Local mode is set to %s", local_mode)
    apple_m_chip = is_apple_m_chip()
    pipeline_session = get_pipeline_session(local_mode)
    return get_config(local_mode, apple_m_chip, pipeline_session)
