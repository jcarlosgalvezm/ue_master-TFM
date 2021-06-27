import pickle
import logging

from ibm_cloud_sdk_core import ApiException
from ibmcloudant.cloudant_v1 import Document

from flask import current_app
from sklearn.pipeline import make_pipeline
from src.vendor.IBM import cloudant, cos
from src.models.build_model import build_model, DEFAULT_PARAMS
from src.features.build_features import get_preprocessing_steps
from src.data.make_dataset import get_dataset

logger = logging.getLogger('app')


def setup_model():
    cloudant_client = cloudant.get_client(
        current_app.config['MODEL_CATALOG_SERVICE_NAME']
        )
    cos_client = cos.get_client()
    bucket = current_app.config['COS_MODEL_STORAGE_BUCKET']
    db = current_app.config['MODEL_CATALOG_DB']

    try:
        cos_client.create_bucket(Bucket=bucket)
    except cos_client.exceptions.BucketAlreadyExists:
        pass

    obj_files = []
    try:
        # Get latest model definition
        model_definition = cloudant_client.get_document(db=db,
            doc_id='ds_jobs', revs=True).get_result()
        for num, rev in enumerate(
                model_definition['_revisions']['ids'][::-1], start=1
                ):
            rev = model_definition['_rev'].split('-', 1)[0]
            fname = f'v{rev}.pkl'
            obj_files.append(fname)
    except ApiException:
        # Doesn't exists documents, we need to create the first one
        doc = Document(id='ds_jobs', **DEFAULT_PARAMS)
        model_definition = cloudant_client.post_document(
            db=db,
            document=doc
            )
        rev = model_definition['rev'].split('-', 1)[0]
        fname = f'v{rev}.pkl'
        obj_files.append(fname)
        try:
            # Delete legacy cos object to mark it as it needs to be trained
            cos_client.delete_object(Bucket=bucket, Key=fname)
        except cos_client.exceptions.NoSuchKey:
            pass

    for fname in obj_files:
        try:
            # Get the model object
            cos_client.get_object(Bucket=bucket, Key=fname)
            # Everything went well
            logger.info(f'Model v{rev} loaded...')
        except cos_client.exceptions.NoSuchKey:
            # Something went wrong we need to create the model object
            X_train, X_test, y_train, y_test = get_dataset()

            # Clean cloudant fields
            for key in ('_id', '_rev', '_revisions'):
                del(model_definition[key])
            model = build_model(**model_definition)
            pipeline = make_pipeline(get_preprocessing_steps(), model)
            pipeline.fit(X_train, y_train)
            cos_client.put_object(Bucket=bucket, Key=fname,
                Body=pickle.dumps(pipeline))
            logger.info(f'Model v{rev} created and loaded...')
