from src.constants import (
    EPOCHS,
    BATCH_SIZE,
    TRAIN_CSV_PATH,
    VALIDATION_CSV_PATH,
    MODEL_FILENAME,
    TRAIN_FILENAME,
)

from sagemaker.workflow.steps import TrainingStep
from sagemaker.inputs import TrainingInput
from sagemaker.tensorflow import TensorFlow


class TensorFlowEstimator:
    def __init__(self, role: str, train_filename: str):
        """
        Initialize the TensorFlowEstimator with the given role and training filename.

        Parameters
        ----------
        role : str
            The IAM role for SageMaker.
        train_filename : str
            The entry point script for training.
        """
        self.role = role
        self.train_filename = train_filename
        self.config = self.get_config()

    def get_config(self):
        """
        Get the configuration settings for the TensorFlow estimator.

        Returns
        -------
        dict
            A dictionary containing the configuration settings.
        """
        return {
            "image": "763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.3.0-cpu-py37-ubuntu18.04",
            "framework_version": "2.3.0",
            "py_version": "py37",
            "instance_type": "ml.m5.large",
            "session": sagemaker.Session(),
        }

    def create_estimator(self) -> TensorFlow:
        """
        Create and return a TensorFlow estimator.

        Returns
        -------
        TensorFlow
            The TensorFlow estimator.
        """
        return TensorFlow(
            base_job_name="training",
            entry_point=self.train_filename,
            hyperparameters={
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "train_csv_path": TRAIN_CSV_PATH,
                "validation_csv_path": VALIDATION_CSV_PATH,
                "model_filename": MODEL_FILENAME,
            },
            metric_definitions=[
                {"Name": "loss", "Regex": "loss: ([0-9\\.]+)"},
                {"Name": "accuracy", "Regex": "accuracy: ([0-9\\.]+)"},
                {"Name": "val_loss", "Regex": "val_loss: ([0-9\\.]+)"},
                {"Name": "val_accuracy", "Regex": "val_accuracy: ([0-9\\.]+)"},
            ],
            image_uri=self.config["image"],
            framework_version=self.config["framework_version"],
            py_version=self.config["py_version"],
            instance_type=self.config["instance_type"],
            instance_count=1,
            disable_profiler=True,
            sagemaker_session=self.config["session"],
            role=self.role,
        )


def main(role: str, train_filename: str):
    """
    Main function to create and configure the TensorFlow estimator.

    Parameters
    ----------
    role : str
        The IAM role for SageMaker.
    train_filename : str
        The entry point script for training.
    """
    estimator_creator = TensorFlowEstimator(role, train_filename)
    estimator = estimator_creator.create_estimator()
    return estimator


# TODO: check preprocessing step, I've missed it in the cohort
train_model_step = TrainingStep(
    name="train-model",
    step_args=estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    ),
    cache_config=cache_config,
)
