import os
import tempfile

import sagemaker
from preprocess_data.preprocessing_step import setup_preprocessing_step
from preprocess_data.preprocessor import preprocess_and_save_data
from utils.aws_infra_config import (
    get_sagemaker_session,
    get_sm_config_with_local_mode,
    get_sagemaker_client,
    get_iam_client,
    get_region,
)
from sagemaker.workflow.pipeline_context import (
    LocalPipelineSession,
    PipelineSession,
)
from sagemaker.workflow.pipeline import Pipeline
from utils import logging

from constants import AWS_ROLE, LOCAL_MODE, IS_APPLE_M_CHIP, DATA_DIR

logger = logging.Logger().get_logger()
local_pipeline_session = LocalPipelineSession()


def main():
    # setup sagemaker
    logger.info("Starting the SageMaker application")

    sagemaker_config = get_sm_config_with_local_mode(
        local_mode=LOCAL_MODE, apple_m_chip=IS_APPLE_M_CHIP
    )
    sagemaker_session = get_sagemaker_session()
    sagemaker_client = get_sagemaker_client()
    iam_client = get_iam_client()
    region = get_region()
    logger.info("Region: %s", region)

    step = setup_preprocessing_step(AWS_ROLE)

    # TODO: move this to separate file
    pipeline = Pipeline(
        name="preprocessing-only",
        steps=[step],
        sagemaker_session=local_pipeline_session,
    )
    logger.info("Create pipeline...")
    pipeline.create(
        role_arn=sagemaker.get_execution_role(), description="local pipeline example"
    )
    logger.info("Pipeline created")


if __name__ == "__main__":
    main()
