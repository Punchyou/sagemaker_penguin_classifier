import os
from sagemaker.workflow.steps import CacheConfig
from sagemaker.workflow.parameters import ParameterString

S3_LOCATION = os.getenv("S3_LOCATION")

# Caching enables reuse of results to speed up workflows and reduce costs.
cache_config = CacheConfig(enable_caching=True, expire_after="50d")

dataset_location = ParameterString(
    name="dataset_location",
    default_value=f"{S3_LOCATION}/data",
)
