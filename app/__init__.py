import pickle
import logging

from ibm_cloud_sdk_core import ApiException
from ibmcloudant.cloudant_v1 import Document

from flask import Flask, current_app
from flask_restx import Api
from sklearn.pipeline import make_pipeline
from src.vendor.IBM import cloudant, cos
from src.models.build_model import build_model, DEFAULT_PARAMS
from src.features.build_features import get_preprocessing_steps
from src.data.make_dataset import get_dataset


api = Api(
    title='DS_JOBS API',
    description='Master Online IA y Data Science 2020-2021 '
    '- TFM Grupo 4'
    )


def setup_model():
    cloudant_client = cloudant.get_client('MODEL_CATALOG')
    cos_client = cos.get_client()
    bucket = current_app.config['COS_BUCKET']

    try:
        cos_client.create_bucket(Bucket=bucket)
    except cos_client.exceptions.BucketAlreadyExists:
        pass

    cloudant.create_db(cloudant_client, 'models')
    cloudant.create_db(cloudant_client, 'predictions')

    model = None

    try:
        model_definition = cloudant_client.get_document(db='models',
            doc_id='ds_jobs').get_result()
        rev = model_definition['_rev'].split('-', 1)[0]
        fname = f'model_v{rev}.pkl'
    except ApiException:
        # Doesn't exists documents, we need to create the first one
        doc = Document(id='ds_jobs', **DEFAULT_PARAMS)
        model_definition = cloudant_client.post_document(
            db='models',
            document=doc
            )
        rev = model_definition['rev'].split('-', 1)[0]
        fname = f'model_v{rev}.pkl'
        try:
            # Delete legacy cos object to mark it as it needs to be trained
            cos_client.delete_object(Bucket=bucket, Key=fname)
        except cos_client.exceptions.NoSuchKey:
            pass

    try:
        model = cos_client.get_object(Bucket=bucket, Key=fname)
        current_app.logger.info(f'Latest model v{rev} loaded...')
    except cos_client.exceptions.NoSuchKey:
        X_train, X_test, y_train, y_test = get_dataset()
        for key in ('_id', '_rev'):
            del(model_definition[key])
        model = build_model(**model_definition)
        pipeline = make_pipeline(get_preprocessing_steps(), model)
        pipeline.fit(X_train, y_train)
        cos_client.put_object(Bucket=bucket, Key=fname,
            Body=pickle.dumps(pipeline))
        current_app.logger.info(f'Latest model v{rev} created and loaded...')


def create_app():
    app = Flask(__name__)

    if app.config['ENV'] == 'development':
        app.config.from_pyfile('config.py')
    else:
        app.config.from_pyfile('config.prod.py')

    api.init_app(app)

    # API Factory
    from app.api_factory import make_namespaces

    # Setup model
    @app.before_first_request
    def warmup():
        with app.app_context():
            setup_model()
            for ns in make_namespaces(api):
                api.add_namespace(ns)

    # Logging
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

    return app
