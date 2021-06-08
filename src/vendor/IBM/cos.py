import ibm_boto3

import os
from ibm_botocore.client import Config
from functools import lru_cache


@lru_cache
def get_client():
    return ibm_boto3.client("s3",
        ibm_api_key_id=os.getenv('COS_API_KEY_ID'),
        ibm_service_instance_id=os.getenv('COS_INSTANCE_CRN'),
        config=Config(signature_version="oauth"),
        endpoint_url=os.getenv('COS_ENDPOINT')
    )
