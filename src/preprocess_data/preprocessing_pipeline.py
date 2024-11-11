from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_definition_config import (
    PipelineDefinitionConfig,
)

from constants import DATASET_LOCATION, SAGEMAKER_SESSION_CONFIG
from preprocess_data_step.preprocessing_step import setup_preprocessing_step

pipeline_definition_config = PipelineDefinitionConfig(
    use_custom_job_prefix=True
)

preprocessing_step = setup_preprocessing_step(role)
preprocessing_pipeline = Pipeline(
    name="session1-pipeline",
    parameters=[DATASET_LOCATION],
    steps=[
        preprocessing_step,
    ],
    pipeline_definition_config=pipeline_definition_config,
    sagemaker_session=SAGEMAKER_SESSION_CONFIG["session"],
)

preprocessing_pipeline.upsert(role_arn=role)
