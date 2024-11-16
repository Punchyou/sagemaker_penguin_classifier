from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import CacheConfig
from constants import (
    DATASET_LOCATION,
    INPUT_DIR,
    MODEL_FILENAME,
    PREPROCESSING_FILENAME,
    S3_LOCATION,
    S3_MODEL_DESTINATION,
    S3_TEST_BASELINE_DATA_DESTINATION,
    S3_TEST_DATA_DESTINATION,
    S3_TRAIN_BASELINE_DATA_DESTINATION,
    S3_TRAIN_DATA_DESTINATION,
    S3_VALIDATION_DATA_DESTINATION,
    SAGEMAKER_SESSION_CONFIG,
    TEST_BASELINE_CSV_PATH,
    TEST_CSV_PATH,
    TRAIN_BASELINE_CSV_PATH,
    TRAIN_CSV_PATH,
    VALIDATION_CSV_PATH,
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
                    # s3 location of the data
                    source=DATASET_LOCATION.default_value,
                    # where the data will be downloaded for development
                    destination=INPUT_DIR,
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source=TRAIN_CSV_PATH,
                    destination=S3_TRAIN_DATA_DESTINATION,
                ),
                ProcessingOutput(
                    output_name="validation",
                    source=VALIDATION_CSV_PATH,
                    destination=S3_VALIDATION_DATA_DESTINATION,
                ),
                ProcessingOutput(
                    output_name="test",
                    source=TEST_CSV_PATH,
                    destination=S3_TEST_DATA_DESTINATION,
                ),
                ProcessingOutput(
                    output_name="model",
                    source=MODEL_FILENAME,
                    destination=S3_MODEL_DESTINATION,
                ),
                ProcessingOutput(
                    output_name="train-baseline",
                    source=TRAIN_BASELINE_CSV_PATH,
                    destination=S3_TRAIN_BASELINE_DATA_DESTINATION,
                ),
                ProcessingOutput(
                    output_name="test-baseline",
                    source=TEST_BASELINE_CSV_PATH,
                    destination=S3_TEST_BASELINE_DATA_DESTINATION,
                ),
            ],
        ),
        cache_config=cache_config,
    )
