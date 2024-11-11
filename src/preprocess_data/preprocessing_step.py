from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import CacheConfig
from constants import (
    DATASET_LOCATION,
    PREPROCESSING_FILENAME,
    S3_LOCATION,
    SAGEMAKER_SESSION_CONFIG,
)
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput


class SKLearnProcessorWrapper:
    def __init__(self, role: str):
        """
        Initialize the SKLearnProcessorWrapper with the given role.

        Parameters
        ----------
        role : str
            The IAM role for SageMaker.
        """
        self.role = role
        self.config = self.get_config()

    def get_config(self):
        """
        Get the configuration settings for the SKLearnProcessor.

        Returns
        -------
        dict
            A dictionary containing the configuration settings.
        """
        return SAGEMAKER_SESSION_CONFIG

    def create_processor(self) -> SKLearnProcessor:
        """
        Create and return an SKLearnProcessor. The SKLearnProcessor handles
        Amazon SageMaker processing.

        Returns
        -------
        SKLearnProcessor
            The SKLearnProcessor.
        """
        return SKLearnProcessor(
            base_job_name="preprocess-data",
            framework_version=self.config["framework_version"],
            instance_type=self.config["instance_type"],
            instance_count=self.config["instance_count"],
            role=self.role,
            sagemaker_session=self.config["session"],
        )


def setup_preprocessing_step(role: str):
    """
    Set up the preprocessing step for the SageMaker workflow.

    Parameters
    ----------
    role : str
        The IAM role for SageMaker.
    """

    # Configure caching
    cache_config = CacheConfig(enable_caching=True, expire_after="15d")

    processor = SKLearnProcessorWrapper(role)
    process_creator = processor.create_processor()

    return ProcessingStep(
        name="preprocess-data",
        step_args=process_creator.run(
            code=str(PREPROCESSING_FILENAME),
            inputs=[
                ProcessingInput(
                    source=DATASET_LOCATION,
                    destination="/opt/ml/processing/input",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/train",
                    destination=f"{S3_LOCATION}/preprocessing/train",
                ),
                ProcessingOutput(
                    output_name="validation",
                    source="/opt/ml/processing/validation",
                    destination=f"{S3_LOCATION}/preprocessing/validation",
                ),
                ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/test",
                    destination=f"{S3_LOCATION}/preprocessing/test",
                ),
                ProcessingOutput(
                    output_name="model",
                    source="/opt/ml/processing/model",
                    destination=f"{S3_LOCATION}/preprocessing/model",
                ),
                ProcessingOutput(
                    output_name="train-baseline",
                    source="/opt/ml/processing/train-baseline",
                    destination=f"{S3_LOCATION}/preprocessing/train-baseline",
                ),
                ProcessingOutput(
                    output_name="test-baseline",
                    source="/opt/ml/processing/test-baseline",
                    destination=f"{S3_LOCATION}/preprocessing/test-baseline",
                ),
            ],
        ),
        cache_config=cache_config,
    )
