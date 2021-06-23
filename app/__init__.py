import logging

from flask import Flask
from flask_restx import Api

from . import bootstraping

api = Api(
    title='DS_JOBS API',
    description='Master Online IA y Data Science 2020-2021 '
    '- TFM Grupo 4'
    )


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
            bootstraping.setup_model()
            for ns in make_namespaces(api):
                api.add_namespace(ns)

    # Logging
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

    return app
