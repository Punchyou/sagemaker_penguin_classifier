from preprocess_data.preprocessor import (
    preprocess_and_save_data,
)
from preprocess_data.preprocessing_step import setup_preprocessing_step
from utils.aws_infra_config import (
    get_sagemaker_session,
    get_sm_config_with_local_mode,
    get_sagemaker_client,
    get_iam_client,
    get_region,
)
import logging

from constants import AWS_ROLE, LOCAL_MODE, IS_APPLE_M_CHIP, DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    # preprocess data and set up preproessing step
    preprocess_and_save_data(data_dir=DATA_DIR)

    setup_preprocessing_step(AWS_ROLE)

    # Define dataset location parameter


if __name__ == "__main__":
    main()
