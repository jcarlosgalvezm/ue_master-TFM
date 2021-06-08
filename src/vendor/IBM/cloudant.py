from ibmcloudant.cloudant_v1 import CloudantV1
from ibm_cloud_sdk_core import ApiException
from functools import lru_cache


@lru_cache
def get_client(service_name):
    return CloudantV1.new_instance(service_name=service_name)


def create_db(client, db_name):
    try:
        return client.put_database(db=db_name)
    except ApiException as e:
        pass
