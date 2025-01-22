from sagemaker import get_execution_role
from preprocess.preprocessor import preprocess_and_save_data
from preprocess.preprocessor_step import run_processing_job
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
from utils import logging_utils

from constants import AWS_ROLE, LOCAL_MODE, IS_APPLE_M_CHIP

logger = logging_utils.Logger().get_logger()
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

    step = run_processing_job()

    # TODO: move this to separate file
    pipeline = Pipeline(
        name="preprocessing-only",
        steps=[step],
        sagemaker_session=local_pipeline_session,
    )
    logger.info("Create pipeline...")
    pipeline.create(role_arn=get_execution_role(), description="local pipeline example")
    logger.info("Pipeline created")


if __name__ == "__main__":
    main()
