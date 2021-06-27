from ibmcloudant.cloudant_v1 import CloudantV1
from functools import lru_cache


@lru_cache
def get_client(service_name):
    return CloudantV1.new_instance(service_name=service_name)
