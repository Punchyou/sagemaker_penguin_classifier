import os
import logging
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
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


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FrameworkProcessorWrapper:
    def __init__(self, role: str):
        """
        Initialize the FrameworkProcessorWrapper with the given role.

        Parameters
        ----------
        role : str
            The IAM role for SageMaker.
        """
        self.role = role
        self.config = self.get_config()

    def get_config(self):
        """
        Get the configuration settings for the FrameworkProcessor.

        Returns
        -------
        dict
            A dictionary containing the configuration settings.
        """
        return SAGEMAKER_SESSION_CONFIG

    def create_processor(self) -> FrameworkProcessor:
        """
        Create and return a FrameworkProcessor. The FrameworkProcessor handles
        Amazon SageMaker processing.

        Returns
        -------
        FrameworkProcessor
            The FrameworkProcessor.
        """
        return FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version="1.2-1",
            role=self.role,
            instance_type=self.config["instance_type"],
            instance_count=1,
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

    # processor = FrameworkProcessor(
    #     estimator_cls=SKLearn,
    #     framework_version="1.2-1",
    #     role=role,
    #     instance_type="ml.m5.large",
    #     instance_count=1,
    # )
    processor = FrameworkProcessorWrapper(role)
    process_creator = processor.create_processor()

    return ProcessingStep(
        name="preprocess-data",
        step_args=process_creator.run(
            code="preprocess_data/preprocessor.py",
        ),
        # source_dir="src/preprocess_data",
        # dependencies=["constants.py"],),
        inputs=[
            ProcessingInput(
                # s3 location of the data
                source=DATASET_LOCATION.default_value,
                # where the data will be downloaded for development
                destination=INPUT_DIR,
            ),
        ],
        #         outputs=[
        #             ProcessingOutput(
        #                 output_name="train",
        #                 source=TRAIN_CSV_PATH,
        #                 destination=S3_TRAIN_DATA_DESTINATION,
        #             ),
        #             ProcessingOutput(
        #                 output_name="validation",
        #                 source=VALIDATION_CSV_PATH,
        #                 destination=S3_VALIDATION_DATA_DESTINATION,
        #             ),
        #             ProcessingOutput(
        #                 output_name="test",
        #                 source=TEST_CSV_PATH,
        #                 destination=S3_TEST_DATA_DESTINATION,
        #             ),
        #             ProcessingOutput(
        #                 output_name="model",
        #                 source=MODEL_FILENAME,
        #                 destination=S3_MODEL_DESTINATION,
        #             ),
        #             ProcessingOutput(
        #                 output_name="train-baseline",
        #                 source=TRAIN_BASELINE_CSV_PATH,
        #                 destination=S3_TRAIN_BASELINE_DATA_DESTINATION,
        #             ),
        #             ProcessingOutput(
        #                 output_name="test-baseline",
        #                 source=TEST_BASELINE_CSV_PATH,
        #                 destination=S3_TEST_BASELINE_DATA_DESTINATION,
        #             ),
        #         ],
        #     ),
        cache_config=cache_config,
    )
